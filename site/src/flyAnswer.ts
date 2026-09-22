import type { FlySide, TurnTrigger } from "./states";
import "@fontsource/inter-tight/latin-500.css";
import "@fontsource/inter-tight/latin-600.css";

export interface LabelPoint { x: number; y: number }

/** Which side each label goes on, and — once the fly has answered — which one is its answer.
 * A `TurnTrigger` satisfies this; a question still in the simulator supplies only `yesSide`. */
export interface LabelSides {
  yesSide: FlySide;
  answer?: TurnTrigger["answer"];
}

/** DOM text stays crisp at phone sizes and inherits no dark-page text colour. */
export function createAnswerLabels(host: HTMLElement) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:absolute;inset:0;pointer-events:none;opacity:0;visibility:hidden;z-index:1";
  const labels = (["yes", "no"] as const).map(answer => {
    const label = document.createElement("span");
    label.textContent = answer.toUpperCase();
    label.dataset.answer = answer;
    label.style.cssText = 'position:absolute;transform:translate(-50%,-50%);font-family:"Inter Tight",sans-serif;line-height:1;letter-spacing:.02em;font-weight:500;white-space:nowrap';
    overlay.appendChild(label);
    return label;
  });
  host.appendChild(overlay);
  return {
    /** `committed` (0..1, from answerTurnAt's `chosen`): both labels look alike until the fly
     * has committed to its heading, then the chosen one lights up. `turn.answer` is absent while
     * the question is still being simulated — both options are on screen from the first second
     * (2026-09-19 design decision), neither of them lit, on the sides the wording already fixes. */
    render(turn: LabelSides | undefined, opacity: number, size: number, project: (side: FlySide) => LabelPoint,
      committed = 1) {
      overlay.style.opacity = String(opacity);
      overlay.style.visibility = turn && opacity > 0 ? "visible" : "hidden";
      if (!turn) return;
      labels.forEach(label => {
        const chosen = turn.answer !== undefined && label.dataset.answer === turn.answer;
        const lit = chosen && committed >= .5;
        const side = label.dataset.answer === "yes" ? turn.yesSide : turn.yesSide === "left" ? "right" : "left";
        const point = project(side);
        const fontSize = Math.max(13, Math.min(22, size * .07));
        // Reserve the text's half-width inside the canvas, including at 150px.
        const margin = fontSize * 1.2 + 4;
        label.style.left = `${Math.max(margin, Math.min(size - margin, point.x * size))}px`;
        label.style.top = `${Math.max(fontSize, Math.min(size - fontSize, point.y * size))}px`;
        label.style.fontSize = `${fontSize}px`;
        label.style.color = lit ? "#3ba6f1" : "#0c0a09";
        label.style.fontWeight = lit ? "600" : "500";
        label.style.opacity = String(chosen ? .5 + .5 * committed : .5 - .2 * committed);
      });
    },
    dispose() { overlay.remove(); },
  };
}
