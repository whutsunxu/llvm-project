# LLVM Codegen Pipeline and GPU Platforms

This document describes where GPU-specific logic lives in LLVM, focusing on the
**codegen stage** (as opposed to mid-end optimization in
`PassBuilderPipelines.cpp`). It is written with **NVPTX / CUDA / Triton** as the
primary example, with brief AMDGPU notes.

**Primary sources**

| File | Role |
|------|------|
| `llvm/lib/Passes/CodeGenPassBuilder.cpp` | Generic codegen pipeline skeleton |
| `llvm/include/llvm/Passes/CodeGenPassBuilder.h` | Hook API (`addIRPasses`, `addInstSelector`, …) |
| `llvm/lib/Target/NVPTX/NVPTXCodeGenPassBuilder.cpp` | NVPTX/CUDA pipeline recipe |
| `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp` | Instruction selection |
| `llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp` | MIR → PTX emission |
| `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp` | AMDGPU EP + codegen hooks |

---

## 1. Big picture: mid-end vs codegen

```
                    MID-END                           CODEGEN (GPU-heavy)
                    ───────                           ───────────────────
MLIR/Triton IR  →   buildPerModuleDefaultPipeline  →  TargetMachine::buildCodeGenPipeline
                    (generic O3 + few EP hooks)         (NVPTX / AMDGPU *CodeGenPassBuilder)
                                                          ↓
                                                    LLVM IR → Machine IR (MIR) → PTX / ISA
```

**Mid-end** (`PassBuilderPipelines.cpp`): target-neutral optimization — inlining,
loop opts, vectorization. GPU targets may register a few extension-point callbacks
(NVPTX `PipelineStartEP`, AMDGPU `VectorizerEndEP`, etc.).

**Codegen** (`CodeGenPassBuilder`): target-specific lowering from LLVM IR to
machine instructions and assembly. This is where **NVPTX** and **AMDGPU**
backends live.

### Triton NVIDIA compile flow

```
make_llir:  llvm.optimize_module(O3)     ← mid-end (mostly generic)
make_ptx:   translate LLVM → PTX         ← codegen (NVPTX backend)
make_cubin: ptxas                         ← external NVIDIA assembler
```

Relevant Triton files:

- `triton/python/src/llvm.cc` — `optimize_module`, `translate_to_asm`
- `triton/third_party/nvidia/backend/compiler.py` — `make_llir`, `make_ptx`, `make_cubin`

---

## 2. Generic codegen pipeline

Entry point: `CodeGenPassBuilder::buildPipeline()` in
`llvm/lib/Passes/CodeGenPassBuilder.cpp`.

Every target (CPU, GPU) follows this skeleton:

| Stage | IR form | What happens |
|-------|---------|--------------|
| **1. IR passes** | LLVM IR | Target lowering prep (`addIRPasses`) |
| **2. Instruction selection** | LLVM IR → MIR | DAG ISel / GlobalISel (`addInstSelector`) |
| **3. Machine SSA opts** | MIR (virtual regs) | Peephole, CSE, scheduling |
| **4. Regalloc prep** | MIR | PHI elim, 2-addr, coalescing |
| **5. Register allocation** | MIR | Assign physical regs (or **skipped on NVPTX**) |
| **6. Post-RA** | MIR | Prologue/epilog, frame indices |
| **7. Late machine opts** | MIR | Post-RA sink, peephole |
| **8. AsmPrinter** | MIR → MC | Emit PTX / AMDGPU asm / object |

Simplified call chain:

```cpp
CodeGenPassBuilder::buildPipeline(MPM, ...)
  addISelPasses(PMW)           // includes addIRPasses + addInstSelector
  addCoreISelPasses(PMW)
  addMachinePasses(PMW)          // SSA opts, regalloc, post-RA, sched
  addAsmPrinter(PMW)             // MIR → assembly text
```

GPU backends **override virtual hooks** rather than forking the entire pipeline:

| Hook | Purpose |
|------|---------|
| `addIRPasses` | Target-specific LLVM IR lowering before ISel |
| `addInstSelector` | LLVM IR → MachineInstr (MIR) |
| `addPreRegAlloc` | Passes immediately before register allocation |
| `addOptimizedRegAlloc` / `addFastRegAlloc` | Regalloc pipeline (customized on NVPTX) |
| `addPostRegAlloc` | Frame indices, prolog/epilog helpers |
| `addAsmPrinter` | Final assembly emission |

---

## 3. GPU backends in LLVM

| Target | Triple | Directory |
|--------|--------|-----------|
| **NVPTX** (CUDA/PTX) | `nvptx64-nvidia-cuda` | `llvm/lib/Target/NVPTX/` |
| **AMDGPU** (ROCm) | `amdgcn-amd-amdhsa` | `llvm/lib/Target/AMDGPU/` |

Typical layout per target:

```
lib/Target/NVPTX/
  NVPTXCodeGenPassBuilder.cpp   ← codegen pipeline recipe
  NVPTXISelDAGToDAG.cpp         ← instruction selection
  NVPTXAsmPrinter.cpp           ← MIR → PTX
  NVVMReflect.cpp               ← mid-end + codegen IR pass
  NVPTXLowerArgs.cpp              ← kernel argument lowering
  *.td                            ← TableGen: instrs, regs, calling conv

lib/Target/AMDGPU/
  AMDGPUTargetMachine.cpp
  AMDGPU* ISel, AsmPrinter, …
```

---

## 4. NVPTX codegen (CUDA / Triton)

Defined in `NVPTXCodeGenPassBuilder.cpp`. Wired via:

```cpp
NVPTXTargetMachine::buildCodeGenPipeline(...)
  → NVPTXCodeGenPassBuilder(...).buildPipeline(...)
```

### 4.1 `addIRPasses` — GPU IR lowering (densest GPU-specific region)

Runs on **LLVM IR** before instruction selection.

| Pass | Role |
|------|------|
| `NVVMReflectPass` | Fold `__nvvm_reflect("__CUDA_ARCH")`, `__CUDA_FTZ` to constants |
| `NVPTXImageOptimizerPass` | Image/texture optimizations (if `-O`) |
| `NVPTXAssignValidGlobalNamesPass` | Sanitize global names for PTX |
| `GenericToNVVMPass` | Lower generic intrinsics to NVVM |
| `ExpandVariadicsPass` | Lower variadic calls before addr-space inference |
| `NVPTXMarkKernelPtrsGlobalPass` | Mark kernel pointers global (CUDA) |
| `NVPTXPromoteParamAlignPass` | Param alignment promotion |
| **`NVPTXLowerArgsPass`** | Rewrite kernel **byval** args to param address space |
| `SROAPass` | Break aggregates after byval lowering |
| `NVPTXLowerAllocaPass` | Lower stack (`.local`) allocas |
| **`InferAddressSpacesPass`** | Propagate global / shared / local / param address spaces |
| `NVPTXAtomicLowerPass` | Lower atomics for GPU |
| Straight-line opts | `SeparateConstOffsetFromGEP`, `SpeculativeExecution`, `SLSR`, GVN/EarlyCSE |
| `AtomicExpandPass` | Expand atomic IR |
| `NVPTXCtorDtorLoweringPass` | Constructor/destructor lowering |
| `Base::addIRPasses` | Generic codegen IR passes (LSR, etc.) |
| `LoadStoreVectorizerPass` | IR-level memory vectorization |
| `NVPTXTagInvariantLoadsPass` | Tag invariant loads |
| `NVPTXIRPeepholePass` | NVPTX-specific IR peepholes |
| `NVPTXLowerUnreachablePass` | WAR ptxas unreachable bugs (some SM versions) |

**Why it matters:** After Triton mid-end opt, codegen IR passes fix address spaces,
kernel signatures, atomics, and NVVM conventions before ISel.

### 4.2 `addInstSelector` — LLVM IR → MIR

| Pass | Role |
|------|------|
| `NVPTXLowerAggrCopiesPass` | Lower aggregate copies |
| `NVPTXAllocaHoistingPass` | Hoist allocas for PTX layout |
| **`NVPTXISelDAGToDAGPass`** | **Main ISel:** SelectionDAG → NVPTX `MachineInstr` |
| `NVPTXReplaceImageHandlesPass` | Replace image handles |

ISel uses **TableGen** patterns in `NVPTXInstrInfo.td`, `NVPTXInstrFormats.td`,
etc., plus C++ custom lowering in `NVPTXISelLowering.cpp`.

### 4.3 Machine passes — MIR (NVPTX-specific behavior)

NVPTX has **no physical register allocation**. Virtual registers map directly to
PTX registers in AsmPrinter:

```cpp
// NVPTXCodeGenPassBuilder constructor
void addTargetRegisterAllocator(PassManagerWrapper &PMW, bool) override {}
```

Many CPU post-RA passes are **explicitly disabled** (tail dup, post-RA sink,
standard prolog/epilog inserter, etc.) because they assume phys regs.

| Hook | NVPTX passes |
|------|--------------|
| `addPreRegAlloc` | `NVPTXForwardParamsPass`, `NVPTXAddressFolderPass`, `NVPTXProxyRegErasurePass` |
| `addOptimizedRegAlloc` | `ProcessImplicitDefs`, `UnreachableMachineBlockElim`, `LiveVariables`, `PHIElimination`, `TwoAddressInstruction`, `RegisterCoalescer`, **`MachineScheduler`** |
| `addPostRegAlloc` | **`NVPTXPrologEpilogPass`** (frame object offsets), **`NVPTXPeepholePass`** (VRFrame → VRFrameLocal) |

### 4.4 `addAsmPrinter` — MIR → PTX

| Pass | Role |
|------|------|
| `NVPTXAsmPrinterBeginPass` | Module-level asm setup |
| **`NVPTXAsmPrinterPass`** | **Emit PTX text:** `.entry`, `.reg`, `.param`, memory spaces, predicates |
| `NVPTXAsmPrinterEndPass` | Module-level asm cleanup |

Output is **PTX assembly** consumed by **ptxas** → CUBIN.

### 4.5 Mid-end NVPTX hooks (before codegen)

Registered in `NVPTXTargetMachine::registerPassBuilderCallbacks` at
**`PipelineStartEP`** (runs before simplification in `buildPerModuleDefaultPipeline`):

| Pass | Role |
|------|------|
| `NVVMReflectPass` | Early reflect folding (if SM known) |
| `NVVMIntrRangePass` | `!range` on `%tid`, `%ntid`, cluster intrinsics |
| `NVPTXCopyByValArgsPass` | Optional early byval copy (`-nvptx-early-byval-copy`, off by default) |

**Triton note:** `llvm.optimize_module` often runs **without** an NVPTX
`TargetMachine`, so these EP passes may **not** run during `make_llir`. Reflect
/ intr-range may run again in **`addIRPasses`** at codegen time.

---

## 5. AMDGPU codegen (brief)

Same `CodeGenPassBuilder` skeleton; different passes and **real** register
allocation.

**Mid-end EP hooks** (`AMDGPUTargetMachine.cpp`):

- `PipelineEarlySimplificationEP` — printf binding, HIP stdpar, always-inline
- `PeepholeEP` — native calls, libcall simplify, uniform intrinsics
- `CGSCCOptimizerLateEP` — promote kernel args, **`InferAddressSpacesPass`**, lower kernel attrs
- `VectorizerEndEP` — **`InferAddressSpacesPass`**
- `OptimizerLastEP` — **`AMDGPUAttributorPass`**

**Codegen:** `AMDGPUCodeGenPassBuilder` — ISel (DAG or GlobalISel), full regalloc,
`AMDGPUAsmPrinter` → GCN assembly.

Key GPU concepts: wavefronts, LDS (shared), buffer resources, kernel calling
convention, PAL metadata.

---

## 6. What is *not* GPU-specific in codegen

Shared infrastructure in `CodeGenPassBuilder.cpp`:

- `PHIEliminationPass`, `TwoAddressInstructionPass`, `RegisterCoalescerPass`
- `MachineSchedulerPass`, `ExpandPostRAPseudosPass`
- `MachineVerifierPass`, MC layer, object/stream emission

Targets enable/disable subsets. NVPTX disables many CPU-oriented passes that
assume prolog/epilog + phys reg lifecycle.

---

## 7. Triton: what runs where

| Step | LLVM component | GPU-specific? |
|------|----------------|---------------|
| MLIR → LLVM IR | Triton lowering (`TritonGPUToLLVM`, NVVM dialect) | Partially |
| `llvm.optimize_module(O3)` | `buildPerModuleDefaultPipeline` | Mostly **no**; Triton adds `BreakStructPhiNodesPass`, disables SLP on sm_80 |
| `ExpandMaskedDivRemPass` | After O3 in `llvm.cc` | NVIDIA-specific |
| PTX emission | **NVPTX `buildCodeGenPipeline`** | **Yes** — full GPU codegen |
| `ptxas` | External tool | NVIDIA |

For GPU debugging after mid-end IR is stable, focus on:

1. **`NVPTXCodeGenPassBuilder::addIRPasses`** — address spaces, kernel args, atomics
2. **`NVPTXLowerArgsPass` / `InferAddressSpacesPass`** — common correctness issues
3. **`NVPTXISelDAGToDAGPass` + `NVPTXISelLowering.cpp`** — how ops become machine instrs
4. **`NVPTXAsmPrinter.cpp`** — final PTX shape
5. **TableGen** — `NVPTXInstrInfo.td`, registers, calling conventions

---

## 8. End-to-end diagram (Triton → PTX)

```
Triton MLIR
    ↓  TritonNVIDIAGPUToLLVM
LLVM IR (+ nvvm.*, llvm inline asm)
    ↓  llvm.optimize_module(O3)          [PassBuilderPipelines — generic mid-end]
Optimized LLVM IR (.llir dump)
    ↓  TargetMachine::buildCodeGenPipeline
      addIRPasses                         [NVVM reflect, lower args, addr spaces, …]
      addInstSelector                     [NVPTXISelDAGToDAG → MIR]
      addMachinePasses                    [sched, NVPTX prolog/epilog/peephole]
      addAsmPrinter                       [NVPTXAsmPrinter → PTX text]
PTX
    ↓  ptxas
CUBIN
```

---

## 9. Suggested reading order

1. `llvm/lib/Passes/CodeGenPassBuilder.cpp` — `buildPipeline()`, `addMachinePasses()`
2. `llvm/lib/Target/NVPTX/NVPTXCodeGenPassBuilder.cpp` — NVPTX overrides
3. `llvm/lib/Target/NVPTX/NVPTXLowerArgs.cpp` — kernel argument / byval lowering
4. `llvm/lib/Target/NVPTX/NVVMReflect.cpp` — SM/FTZ conditional compilation
5. `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp` — instruction selection
6. `llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp` — PTX emission
7. `llvm/lib/Target/NVPTX/NVPTXInstrInfo.td` — instruction patterns (TableGen)

Related mid-end doc: `workspace/pass_builder_pipelines.md`.

---

## 10. Summary

| Layer | GPU relevance |
|-------|----------------|
| **Mid-end** (`PassBuilderPipelines`) | Mostly generic; few EP hooks (NVPTX reflect, AMDGPU addr spaces) |
| **Codegen IR** (`addIRPasses`) | **High** — NVVM, address spaces, kernel args, atomics |
| **ISel** (`addInstSelector`) | **High** — GPU instruction patterns, calling conventions |
| **Machine passes** | **Medium** — NVPTX skips regalloc; scheduling/peephole still matter |
| **AsmPrinter** | **High** — PTX / GCN text is the backend output |

**One line:** GPU platform logic in LLVM codegen lives under **`lib/Target/NVPTX/`**
(or **`AMDGPU/`**): IR lowering in **`addIRPasses`**, selection in **`addInstSelector`**, and assembly in **`AsmPrinter`**. Mid-end optimization is shared; codegen is where CUDA kernels become PTX.
