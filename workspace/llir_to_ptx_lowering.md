# LLIR → MIR → PTX: Terminology and Lowering Path (Triton / NVPTX)

This document explains the IR tiers used in Triton’s NVIDIA backend, and how
optimized **LLIR** is lowered to **PTX** (assembly) and then **CUBIN**, with key
functions and file locations woven into each stage.

**Related docs**

- `workspace/pass_builder_pipelines.md` — mid-end (`optimize_module` / PassBuilder)
- `workspace/codegen_pipeline.md` — codegen overview and GPU backends

---

## 1. Terminology and tiers

Think of **layers of representation**, not different assemblers.

```
Source / high IR (MLIR / TTGIR)
    ↓
LLVM IR  (often called LLIR in Triton)
    ↓
MIR      (Machine IR — still inside LLVM)
    ↓
PTX      (= NVPTX “asm” — text LLVM emits)
    ↓
CUBIN    (binary GPU code — after ptxas)
```

| Term | What it is | Who owns it | Looks like |
|------|------------|-------------|------------|
| **LLIR** | **L**LVM **IR** (Triton dump name). Target-independent-ish SSA IR after MLIR→LLVM and `optimize_module`. | LLVM mid-end | `%1 = add i32 %a, %b`, `call @llvm.nvvm…` |
| **MIR** | **M**achine **IR**. After instruction selection: ops are target instructions, still in LLVM’s codegen. | LLVM codegen | `ADD_rri`, `LD_global`, virtual regs |
| **ASM / PTX** | For NVPTX, **asm = PTX**. Text assembly LLVM’s AsmPrinter prints. Not x86 asm. | LLVM AsmPrinter → string | `.visible .entry …`, `add.s32 %r1, %r2, %r3;` |
| **CUBIN** | GPU **binary** object. Not produced by LLVM in Triton; **ptxas** assembles PTX. | NVIDIA `ptxas` | ELF-like `.cubin` |

### Common confusions

- **LLIR vs LLVM IR** — same thing; Triton dumps call it `*.llir`.
- **MIR vs PTX** — MIR is in-memory machine instructions; PTX is printed text.
- **ASM vs PTX** — on NVPTX, asm means PTX. `translate_to_asm` returns a PTX string.
- **Is CUBIN asm?** — no. CUBIN is machine code. PTX is the asm; ptxas is the assembler.

### Triton NVIDIA stack

```
TTTT / TTGIR (MLIR)          ← Triton dialects
    ↓  make_llir
LLIR (LLVM IR)               ← after optimize_module  [mid-end]
    ↓  make_ptx / translate_to_asm
MIR (inside LLVM, temporary) ← ISel + machine passes
    ↓  NVPTXAsmPrinter
PTX (text asm)               ← what you see as "asm"
    ↓  make_cubin / ptxas
CUBIN (binary)               ← what GPU loads
```

**Important:** Triton’s `make_ptx` uses the **legacy** codegen path
(`TargetMachine::addPassesToEmitFile` → `NVPTXPassConfig`), not
`CodeGenPassBuilder::buildPipeline`. The two pipelines are intentionally mirrored
(`NVPTXCodeGenPassBuilder.cpp` must stay in sync with `NVPTXPassConfig`).

---

## 2. Lowering from LLIR to asm (PTX), with function locations

Starting from **optimized LLIR** (after `make_llir` / `llvm.optimize_module`).

### Stage 0 — Python entry

Optimized LLIR arrives as a string in `CUDABackend.make_ptx` in
`triton/third_party/nvidia/backend/compiler.py` (around line 575). That function
builds `triple`, `proc`, `features`, and `flags`, then calls
`llvm.translate_to_asm(...)` at line 589 with `isObject=False`, so LLVM must emit
**assembly text** (PTX), not an object file.

Typical arguments (example for capability 90):

- `triple` = `"nvptx64-nvidia-cuda"`
- `proc` = `"sm_90a"` (`sm_arch_from_capability`)
- `features` = `"+ptxN"` (`get_features`)
- `flags` = `["nvptx-mad-wide-opt"]`
- `enable_fp_fusion` = `opt.enable_fp_fusion` (often `True`)
- `isObject` = `False`
- `canonicalize_gep` = `"fpsan" in opt.instrumentation_mode`

---

### Stage 1 — Bind and prepare the module (still LLIR)

The Python binding is registered in `init_triton_llvm` as `translate_to_asm` in
`triton/python/src/llvm.cc` (around line 879). That lambda parses the LLIR string
with `llvm::parseIR`, then calls **`translateLLVMIRToASM`** in the same file
(around line 389).

Inside `translateLLVMIRToASM`, **`createTargetMachine`** (same file, around line
190) builds an NVPTX `TargetMachine` from the triple/proc/features (e.g.
`nvptx64-nvidia-cuda`, `sm_90a`, `+ptx…`, usually `CodeGenOptLevel::Aggressive`).
The module still holds **LLVM IR**. Before codegen, a small legacy pass manager
runs AlwaysInliner and Verifier on that module (same function, roughly lines
423–440). Optional GEP canonicalization may run if `canonicalizeGEP` is true.

**Form:** `llvm::Module` (LLIR).

---

### Stage 2 — Kick off codegen

Still in `translateLLVMIRToASM`, emission starts with:

```cpp
machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
```

(`llvm.cc` around line 470), with `fileType = AssemblyFile`.

That lands in **`CodeGenTargetMachineImpl::addPassesToEmitFile`** in
`llvm/lib/CodeGen/CodeGenTargetMachineImpl.cpp` at line **232**. That function
does two things:

1. **`addPassesToGenerateCode`** (same file, line **118**) — fill the pass
   manager with IR→MIR codegen passes.
2. **`addAsmPrinter`** (same file, line **145**) — append the printer that turns
   MIR into PTX text.

`addPassesToGenerateCode` asks the target for a pass config via
**`NVPTXTargetMachine::createPassConfig`** in
`llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp` (line **166**), which returns
**`NVPTXPassConfig`** (defined in the same file around line **126**). Then it
calls `PassConfig->addISelPasses()` and `PassConfig->addMachinePasses()`.

---

### Stage 3a — LLIR prep (still LLIR)

The ISel pipeline is rooted in **`TargetPassConfig::addISelPasses`** in
`llvm/lib/CodeGen/TargetPassConfig.cpp` at line **1079**. Early in that function
it calls **`addIRPasses()`**, which for NVPTX is overridden by
**`NVPTXPassConfig::addIRPasses`** in `NVPTXTargetMachine.cpp` at line **245**.

That is where GPU IR cleanup runs while the IR is still LLVM IR:

- NVVM reflect (`createNVVMReflectPass` / `NVVMReflect.cpp`) — fold
  `__CUDA_ARCH` / FTZ conditionals
- `GenericToNVVM`
- **`NVPTXLowerArgs`** (`NVPTXLowerArgs.cpp`) — kernel byval → param AS
- Address-space inference via **`addAddressSpaceInferencePasses`** (same file,
  around line **216**, including `InferAddressSpaces`, `NVPTXLowerAlloca`,
  `NVPTXAtomicLower`)
- `AtomicExpand`, ctor/dtor lowering
- Generic `TargetPassConfig::addIRPasses` (LSR, etc.)
- `LoadStoreVectorizer`, `NVPTXIRPeephole`, optional `NVPTXLowerUnreachable`

After `addIRPasses`, `addISelPasses` continues with **`addCodeGenPrepare`**,
exception handling, and **`addISelPrepare`** (still in `TargetPassConfig.cpp`,
around lines 1095–1097).

**Form:** cleaned LLIR. No MIR yet.

---

### How 3a goes to 3b

There is no special jump. **`addISelPasses`** finishes IR prep, then immediately
calls **`addCoreISelPasses`** (end of `TargetPassConfig.cpp` line 1079’s
function, return at line 1099). At run time the legacy pass manager runs all 3a
FunctionPasses on each function, creates an empty `MachineFunction` via
`MachineModuleInfo`, then runs the ISel MachineFunctionPass — that pass *is* the
boundary.

---

### Stage 3b — LLIR → MIR (instruction selection) ★

At the end of `addISelPasses`, control goes to
**`TargetPassConfig::addCoreISelPasses`** in `TargetPassConfig.cpp` at line
**991**. For NVPTX that calls **`NVPTXPassConfig::addInstSelector`** in
`NVPTXTargetMachine.cpp` at line **332**, which adds LowerAggrCopies,
AllocaHoisting, then the real selector:

**`createNVPTXISelDag`** in `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp`
(around line **177**), implemented by **`NVPTXDAGToDAGISel`** (subclass of
`SelectionDAGISel` in the same file).

At run time the handoff is
**`SelectionDAGISel::runOnMachineFunction`** in
`llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp` at line **605**. It reads
`mf.getFunction()` (the prepared LLIR), then **`SelectAllBasicBlocks`** (line
**1659**) walks LLVM basic blocks, builds a SelectionDAG, and
**`CodeGenAndEmitDAG`** (around line **947**) legalizes/matches patterns and
emits **`MachineInstr`s** into the `MachineFunction`. Patterns and custom
lowering live in `NVPTXISelDAGToDAG.cpp`, `NVPTXISelLowering.cpp`, and
`NVPTXInstrInfo.td`.

Conceptual mapping:

```
LLVM Instruction          SelectionDAG           MachineInstr (MIR)
─────────────────         ────────────           ──────────────────
%v = add i32 %a,%b   →    add node          →    ADD…  %vreg1, %vreg2, %vreg3
%load = load …       →    load node         →    LD_global…
call @llvm.nvvm…     →    intrinsic node    →    matching NVPTX opcode
```

After ISel, `FinalizeISel` runs.

**Form after this:** MIR (`MachineFunction` / `MachineInstr`).

---

### Stage 3c — MIR machine passes

After ISel, **`TargetPassConfig::addMachinePasses`** in `TargetPassConfig.cpp`
at line **1127** runs the machine pipeline. NVPTX injects:

- **`NVPTXPassConfig::addPreRegAlloc`** (`NVPTXTargetMachine.cpp` line **341**) —
  forward params, address fold, proxy-reg erase
- **`addOptimizedRegAlloc`** (line **368**) — PHI elim, two-address, coalescer,
  MachineScheduler; **`createTargetRegisterAllocator`** returns `nullptr`
  (line **359**), so there is **no physical register allocation** (PTX uses
  virtual regs end-to-end)
- **`addPostRegAlloc`** (line **349**) — **`NVPTXPrologEpilog`** and
  **`NVPTXPeephole`**

Many CPU-oriented post-RA passes are disabled in `addIRPasses` (prolog/epilog
inserter, tail dup, post-RA sink, etc.) because they assume phys regs.

**Form:** MIR (scheduled / frame-adjusted).

---

### Stage 4 — MIR → PTX (this is “asm”) ★

Back in `addPassesToEmitFile`, after the codegen passes are installed,
**`CodeGenTargetMachineImpl::addAsmPrinter`**
(`CodeGenTargetMachineImpl.cpp` line **145**) creates an MC asm streamer and the
target printer. For NVPTX that is **`NVPTXAsmPrinter`** in
`llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp`.

Per function, **`NVPTXAsmPrinter::runOnMachineFunction`** (around line **923**)
walks MIR; each instruction is printed by
**`NVPTXAsmPrinter::emitInstruction`** (around line **582**) into the stream
Triton captured. The result is a **PTX string** — for NVPTX, that *is* the
assembly.

`translateLLVMIRToASM` returns that string to Python.

---

### Stage 5 — Post-process PTX

Still in **`make_ptx`** (`compiler.py` around lines 595–597), regex rewrites
`.version` and `.target sm_N`, and may strip debug flags. Still PTX text.

---

### Stage 6 — Beyond LLVM

**`make_cubin`** in `compiler.py` (around line **609**) shells out to **`ptxas`**,
which assembles PTX into **CUBIN**. That is outside LLVM’s AsmPrinter path.

---

## 3. End-to-end flow with locations

```
make_ptx                                      compiler.py:575
  → translate_to_asm / translateLLVMIRToASM   llvm.cc:879 / 389
       createTargetMachine                    llvm.cc:190
  → addPassesToEmitFile                       CodeGenTargetMachineImpl.cpp:232
       → addPassesToGenerateCode              CodeGenTargetMachineImpl.cpp:118
            createPassConfig → NVPTXPassConfig  NVPTXTargetMachine.cpp:166
            addISelPasses                     TargetPassConfig.cpp:1079
              addIRPasses                     NVPTXTargetMachine.cpp:245   [3a LLIR]
              addCoreISelPasses               TargetPassConfig.cpp:991
                addInstSelector               NVPTXTargetMachine.cpp:332
                createNVPTXISelDag            NVPTXISelDAGToDAG.cpp:177
                runOnMachineFunction          SelectionDAGISel.cpp:605     [3b MIR]
                SelectAllBasicBlocks          SelectionDAGISel.cpp:1659
            addMachinePasses                  TargetPassConfig.cpp:1127     [3c MIR]
              addPre/PostRegAlloc             NVPTXTargetMachine.cpp:341/349
       → addAsmPrinter                        CodeGenTargetMachineImpl.cpp:145
            NVPTXAsmPrinter::emitInstruction  NVPTXAsmPrinter.cpp:582       [4 PTX]
  → make_ptx post-process                     compiler.py:595
  → make_cubin / ptxas                        compiler.py:609               [CUBIN]
```

---

## 4. Representation changes (only two inside LLVM)

```
 LLIR ──(IR passes)──► LLIR' ──(ISel)──► MIR ──(machine passes)──► MIR' ──(AsmPrinter)──► PTX
  ↑                      ↑                 ↑                                              ↑
optimize_module      LowerArgs/AS      NVPTXISelDag                                  "asm"
(make_llir)          (codegen 3a)      (3b)                                        (make_ptx)
                                                                                         │
                                                                                      ptxas
                                                                                         ↓
                                                                                      CUBIN
```

| Boundary | Key function | Location |
|----------|--------------|----------|
| LLIR → MIR | `SelectionDAGISel::runOnMachineFunction` / `createNVPTXISelDag` | `SelectionDAGISel.cpp:605`, `NVPTXISelDAGToDAG.cpp:177` |
| MIR → PTX | `NVPTXAsmPrinter::emitInstruction` | `NVPTXAsmPrinter.cpp:582` |

---

## 5. Legacy vs New Pass Manager note

| Path | Entry | Used by Triton `make_ptx`? |
|------|-------|----------------------------|
| **Legacy** | `addPassesToEmitFile` → `NVPTXPassConfig` | **Yes** |
| **New PM** | `buildCodeGenPipeline` → `NVPTXCodeGenPassBuilder::buildPipeline` | **No** |

Same NVPTX codegen *content* (IR lowering, ISel, AsmPrinter); different pass-manager
API. Reading `NVPTXCodeGenPassBuilder.cpp` still describes what Triton’s legacy
path does, because they are intentionally kept in sync.
