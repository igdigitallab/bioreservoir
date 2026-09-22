// What the visitor sees during the ~30 s between "Ask" and the answer. Every stage is a real step
// of the pipeline (live/worker.py): the queue, the text encoder, the 250 ms whole-brain
// simulation, the left/right readout. The only moving number is honest elapsed wall time, never a
// made-up progress percentage: the simulation reports no progress until it ends.
import { h, clear } from "../dom";

export type WaitPhase = { kind: "queued"; position: number } | { kind: "thinking"; elapsedS: number };

export interface StageLine {
  label: string;
  state: "done" | "active" | "todo";
}

// Encoding the question takes well under a second of the thinking phase; after that the
// simulation is what the wall clock is spent on.
const ENCODE_S = 1;

export function stageLines(phase: WaitPhase): StageLine[] {
  const queued = phase.kind === "queued";
  const elapsed = phase.kind === "thinking" ? phase.elapsedS : 0;
  const encoding = !queued && elapsed < ENCODE_S;
  return [
    { label: queued ? `In the queue: #${phase.position}` : "In the queue", state: queued ? "active" : "done" },
    { label: "Your words become a pattern on the fly's sensory neurons", state: queued ? "todo" : encoding ? "active" : "done" },
    {
      label: `165,000 neurons fire for 250 ms of fly time, about 30 s on our CPU${queued ? "" : ` · ${Math.floor(elapsed)} s`}`,
      state: queued || encoding ? "todo" : "active",
    },
    { label: "Reading which way it turned", state: "todo" },
  ];
}

/** A self-updating stage list; `set()` switches phase, `stop()` clears it and its timer. */
export function mountWaitStages(host: HTMLElement): { set: (phase: "queued" | "thinking", position?: number) => void; stop: () => void } {
  let timer: number | undefined;
  let thinkingSince = 0;
  let position = 0;

  function render(phase: WaitPhase) {
    clear(host);
    host.appendChild(
      h(
        "ol",
        { class: "wait-stages" },
        // Only the stages reached so far: they appear one by one as the work gets there
        // (2026-09-19 design decision), instead of the whole list being on screen from the start.
        stageLines(phase)
          .filter((line) => line.state !== "todo")
          .map((line) => h("li", { class: `wait-stage wait-stage-${line.state} reveal-in` }, line.label)),
      ),
    );
  }

  function stop() {
    if (timer !== undefined) window.clearInterval(timer);
    timer = undefined;
    clear(host);
  }

  function set(phase: "queued" | "thinking", pos?: number) {
    if (phase === "queued") {
      if (pos !== undefined) position = pos;
      if (timer !== undefined) window.clearInterval(timer);
      timer = undefined;
      render({ kind: "queued", position });
      return;
    }
    if (timer === undefined) {
      thinkingSince = performance.now();
      timer = window.setInterval(() => render({ kind: "thinking", elapsedS: (performance.now() - thinkingSince) / 1000 }), 1000);
    }
    render({ kind: "thinking", elapsedS: (performance.now() - thinkingSince) / 1000 });
  }

  return { set, stop };
}
