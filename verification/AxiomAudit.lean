import ProtectedDoRA
import Lean

/-!
Audit every theorem declared in a `ProtectedDoRA.*` module (none can be left
off a hand-written list): print the logical axioms each one depends on, and
fail if any depends on something beyond Lean's three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`). An unfinished proof would show
up here as `sorryAx`; a custom assumption as its own name.
-/

open Lean Elab Command

elab "#audit_protected_dora" : command => do
  let env ← getEnv
  let standard : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let mut names : Array Name := #[]
  for (name, info) in env.constants.map₁.toList do
    if let .thmInfo _ := info then
      if let some idx := env.getModuleIdxFor? name then
        let mod := env.header.moduleNames[idx.toNat]!
        if (`ProtectedDoRA).isPrefixOf mod && !name.isInternalDetail then
          names := names.push name
  names := names.qsort (fun a b => a.toString < b.toString)
  let mut failures : Nat := 0
  for name in names do
    let axs ← collectAxioms name
    logInfo m!"{name} depends on axioms: {axs.toList}"
    let extra := axs.toList.filter (· ∉ standard)
    unless extra.isEmpty do
      failures := failures + 1
      logError m!"{name} depends on non-standard axioms: {extra}"
  if names.isEmpty then
    logError "no ProtectedDoRA theorems found"
  else if failures == 0 then
    logInfo m!"audited {names.size} theorems: all use only propext, Classical.choice, Quot.sound"

#audit_protected_dora
