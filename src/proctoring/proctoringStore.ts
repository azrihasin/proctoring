import { create } from 'zustand'

// Lifecycle of a face-mismatch incident.
// CLEAR    -> no active mismatch
// DETECTED -> incident just opened (first tick after the grace period)
// PERSISTING -> mismatch still ongoing on subsequent ticks
// RESOLVED -> incident just closed (transient, immediately followed by CLEAR)
export type IncidentState = 'CLEAR' | 'DETECTED' | 'PERSISTING' | 'RESOLVED'

export interface ProctoringState {
  incidentState: IncidentState
  currentScore: number | null
  incidentId: string | null
  snapshotCount: number
  setIncidentState: (state: IncidentState) => void
  setCurrentScore: (score: number | null) => void
  setIncidentId: (incidentId: string | null) => void
  incrementSnapshotCount: () => void
  reset: () => void
}

export const useProctoringStore = create<ProctoringState>((set) => ({
  incidentState: 'CLEAR',
  currentScore: null,
  incidentId: null,
  snapshotCount: 0,
  setIncidentState: (incidentState) => set({ incidentState }),
  setCurrentScore: (currentScore) => set({ currentScore }),
  setIncidentId: (incidentId) => set({ incidentId }),
  incrementSnapshotCount: () => set((s) => ({ snapshotCount: s.snapshotCount + 1 })),
  reset: () =>
    set({ incidentState: 'CLEAR', currentScore: null, incidentId: null, snapshotCount: 0 }),
}))
