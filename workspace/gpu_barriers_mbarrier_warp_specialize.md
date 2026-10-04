# GPU barriers, mbarrier, and warp specialization (CTA / cluster / warp)

Notes distilled from reading the FP8 warp-specialized matmul LLIR
(`21_after_llvm_optimize_module_O3.llir___p_matmul_…`), focused on how
threads rendezvous across warps and why wrong barrier use hangs.

---

## 1. General idea

On an NVIDIA GPU, warps run independently. **Nothing** orders shared-memory
publishes between warps unless you insert an explicit sync. “Handshake” means:
two (or more) thread groups **meet** so that what one wrote is safe for the
other to read.

Three different mechanisms are easy to confuse:

| Mechanism | Scope | Identity | State lives in |
|-----------|--------|----------|----------------|
| **Named barrier** (`bar.sync` / `barrier.sync`) | CTA (or a thread subset) | Integer **name** `0…15` | **HW slots** (16 per CTA) |
| **`mbarrier`** | CTA (`shared::cta`) or cluster (`shared::cluster`) | **Pointer** to an 8-byte object | **Shared memory** (as many as you allocate) |
| **Warp sync** (`bar.warp.sync`, `elect.sync`, …) | One warp (32 lanes) | Lane **mask** | Warp HW |

Cluster adds another layer (`barrier.cluster`, cluster-scoped mbarrier /
DSMEM). This matmul is mostly **CTA-local**.

**IR line number does not create a barrier.** Hardware keys only on
**name** (named barrier) or **address** (mbarrier). Two different basic
blocks can participate in the **same** rendezvous if they execute the same
name (or touch the same mbarrier object) in the same episode.

---

## 2. Named barriers (`bar.sync`) — details and typical use

### 2.1 Instruction shape

```ptx
bar.sync  a        ; name a, all threads in CTA (classic)
bar.sync  a, b     ; name a, wait until b threads have arrived
```

LLVM:

```llvm
; Full CTA
call void @llvm.nvvm.barrier.cta.sync.all(i32 %name)

; Subset of the CTA
call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %name, i32 %count)
```

Two **independent** operands:

| Operand | Values | Meaning |
|---------|--------|---------|
| **name** | `0 … 15` | Which of the 16 HW barrier slots |
| **count** | multiple of 32, ≤ CTA size | How many arrives this episode needs |

There is **no** numeric link between “name 1” and “128 threads.”  
`count(0, 128)` means: slot **0**, need **128** arrives — not “barrier 128.”

### 2.2 Typical usage

**Full CTA rendezvous (mailbox / phase change)**  
Default warps write a shared task table; worker warps read it. Both sides
execute `barrier.sync 1` (possibly from different CFG blocks), then proceed.

**Warp-specialize / warpgroup subset**  
Only the threads that share a phase of work sync, e.g.:

- 4 default warps → `count = 128`
- 2-warp partition → `count = 64`

So the rest of the CTA is not forced to stop at every micro-step.

**Mental model**

```text
16 HW doorbells:  [0] [1] [2] … [15]   ← name selects the doorbell
Each doorbell:    "open when N threads have pressed it"  ← count is N
```

### 2.3 Example from the matmul kernel

CTA: 256 threads = 8 warps.

| Name | Count / form | Who | Why |
|------|----------------|-----|-----|
| **1** | `.all` (256) | Warps 0–3 **and** 4–7 | WS mailbox handshake |
| **0** | `128` | Default / producer path (warps 0–3) | Sync among that 128 only |
| **2** | `64` | One 2-warp worker partition | Sync inside that partition |

Mailbox writes (partition IDs `0`/`1` at `smem+99392…`) sit **before**
`barrier.sync 1` on the default path; workers `barrier.sync 1` in
`.preheader` **before** loading those bytes.

---

## 3. `bar.sync.all` vs `bar.sync.count` — usage and failure modes

### 3.1 `.all` — typical use

```llvm
; Default path (%132), after writing mailbox
call void @llvm.nvvm.barrier.cta.sync.all(i32 1)

; Worker path (.preheader), before reading mailbox
call void @llvm.nvvm.barrier.cta.sync.all(i32 1)
```

Meaning: **every thread in the CTA** must arrive at **name 1**.  
CFG position may differ; the name must match for that episode.

**Correct split-path handshake**

```text
warps 0–3  ──►  %132  ──►  store mailbox ──►  barrier.sync 1
warps 4–7  ──►  .preheader ────────────────►  barrier.sync 1
                    └── same HW name 1 ──┘
```

### 3.2 `.count` — typical use

```llvm
; After tensormap publish on producer path only
call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128)
```

Meaning: only **128** arrives needed on **name 0**. Intended participants
are the 128 threads on that path. Workers should be hitting **other names**
(`1`, `2`) or mbarriers at that moment — **not** name `0`.

### 3.3 Problems (with examples)

**Problem A — `.all` with only one CFG side**

```text
Only warps 0–3 execute barrier.sync 1
Warps 4–7 never hit name 1
→ hang (waiting for 256 arrives that never come)
```

This is why “block `%132` only runs warps 0–3” does **not** make line
`sync.all(1)` wrong: workers must (and do) hit the same name elsewhere.

**Problem B — wrong group hits the same name (pollution)**

Hardware does **not** care which IR line ran. Same name + overlapping
execution ⇒ **same arrive pool**.

```text
Episode: producers do  count(0, 128)
Meanwhile workers also execute  count(0, 64)   // BUG
→ arrivals mix on slot 0 → hang or undefined behavior
```

Compiler contract for each episode: **exactly** the intended threads hit
that name; nobody else.

**Problem C — count does not match who actually arrives**

```text
bar.sync 2, 64     // expects 64
but only 32 threads execute it
→ hang
```

Or: 128 threads arrive while count says 64 → also wrong / UB.

**Problem D — reusing a name too early**

The same name may be used again **after** the previous episode completes
(many later `count(0, 128)` in the kernel). That is fine. Overlapping two
logical syncs on one name is not.

**Problem E — confusing name with count**

`count(0, 128)` is **not** “barrier number 128.”  
Name `0` is the slot; `128` is the arrive target.

---

## 4. mbarrier — cross-warp async handshake

Named `bar.sync` answers: “have these **threads** reached this point?”

**`mbarrier`** answers: “has this **async work / tile phase** completed?”

```ptx
mbarrier.init.shared::cta.b64 [smem+off], count;
mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], nbytes;
cp.async.bulk.tensor... mbarrier::complete_tx::bytes ... [bar];
mbarrier.try_wait.parity.shared::cta.b64 complete, [bar], phase;
```

- Object is an **8-byte** record in **shared memory** (e.g. `99328`, `99336`, …).
- **Not** one of the 16 named HW slots.
- Count often `1`: TMA `complete_tx` (or one producer arrive) completes it.
- Typical pattern: **empty/full** barriers × **pipeline stages** (A/B tiles,
  ping-pong). Producer (TMA warps) and consumer (MMA warps) handshake through
  these objects without a full-CTA `bar.sync` on every tile.

In the matmul dump: eight inits at `99328…99384` = four roles × two stages;
later `try_wait` / `expect_tx` / `complete_tx` use those addresses.

---

## 5. Warp specialization glue (how the pieces fit)

```text
Entry: warp_id = tid.x >> 5
  warp_id < 4  →  %132   (default: tensormap, mbarrier init, mailbox, MMA path)
  else         →  .preheader  (workers: read mailbox → partition 0 or 1)

Default:  mbarrier.init ×8
          store task IDs at smem+99392..99395  (0,0,1,1)
          barrier.sync 1          ←──┐
Workers:  barrier.sync 1          ←──┘  full-CTA handshake
          load ID → switch to partition
          bar.sync 2, 64 / mbarrier wait-arrive for tiles
```

- **`elect.sync` + warp0**: only one lane runs `mbarrier.init` (uniform for ptxas).
- **`setmaxnreg`**: register budget split between lean producers and fat consumers
  (orthogonal to barriers, but part of the same WS split).

---

## 6. CTA vs cluster (short)

| | CTA | Cluster |
|--|-----|---------|
| Who | Threads in one block | Threads across CTAs in a cluster |
| Named barrier | `bar.sync` / `barrier.sync` (16 slots) | Not the same 16-slot CTA table |
| mbarrier | `shared::cta` | `shared::cluster` / DSMEM; cross-CTA arrive rules differ |
| This kernel | Primary sync domain | Little / no cluster sync in the studied path |

---

## 7. Cheat sheet

| Need | Prefer |
|------|--------|
| Whole CTA sees shared mailbox / args | `barrier.sync name` (`.all`) from **all** sides |
| Only N threads in a WS group | `barrier.sync name, N` (`.count`) |
| Tile ready / free with TMA | `mbarrier` + `expect_tx` / `complete_tx` / `try_wait` |
| Align lanes in one warp | `bar.warp.sync` / `elect.sync` |
| Avoid hang | Same name ⇒ same episode; count matches actual arrivers; don’t mix unrelated groups on one name |

**One line:** named barriers are 16 CTA doorbells (name + optional count);
mbarriers are smem objects for async producer/consumer; warp-specialize
handshakes use full-CTA `bar.sync` for mailbox publish and subset
`bar.sync` / mbarrier for the pipeline — IR location is irrelevant, **name
and who actually arrives** are everything.
