import { createFlyScene } from "../src/flyScene";
import { createFlyIcon } from "../src/flyIcon";
import { createFlyIcon as createFlyIcon2d } from "../src/flyIcon2d";
import { answerTurnAt, ANSWER_HOLD_SECONDS, ANSWER_RELEASE_SECONDS, driveAt, flurryDuration, flurryStart, IDLE_SEED, poseAt, triggerDrive } from "../src/flyPose";
import { createAnswerLabels } from "../src/flyAnswer";
import type { AnimationTrigger, TurnTrigger } from "../src/states";

const params = new URLSearchParams(location.search);
const animation = params.get("anim") || "idle";
const size = Math.max(100, Math.min(480, Number(params.get("size")) || 320));
const seed = Number(params.get("seed") ?? IDLE_SEED);
const host = document.querySelector<HTMLElement>("#specimen")!;
host.style.width = `${size}px`;
host.style.height = `${size}px`;
document.querySelector("#caption")!.textContent = `${animation.replaceAll("-", " ")} · ${size} × ${size}`;
const names = ["idle", "answer", "turn-left", "turn-right", "appetite", "fear", "backoff", "courtship", "arousal"];
for (const name of names) {
  const link = document.createElement("a");
  link.href = `?anim=${name}&t=3&size=${size}&seed=${seed}${params.has("reduced") ? "&reduced" : ""}`;
  link.textContent = name.replaceAll("-", " ");
  if (name === animation) link.setAttribute("aria-current", "page");
  document.querySelector("#controls")!.appendChild(link);
}
const answer = params.get("answer") === "no" ? "no" : "yes";
const yesSide = params.get("yesSide") === "right" ? "right" : "left";
const turn: TurnTrigger = {
  kind: "turn", answer, yesSide,
  toward: answer === "yes" ? yesSide : yesSide === "left" ? "right" : "left",
  intensity: Math.abs(Number(params.get("bias") ?? .01)),
};
const triggers: AnimationTrigger[] = animation === "idle" ? [] :
  animation === "answer" ? [turn] : animation.startsWith("turn-") ?
    [{ ...turn, answer: "yes", yesSide: animation === "turn-left" ? "left" : "right",
      toward: animation === "turn-left" ? "left" : "right", intensity: 1 }] :
    [{ kind: animation as Exclude<AnimationTrigger["kind"], "turn">, intensity: 1 }];
const durationMs = animation === "answer" ? 2500 : 6000;
const slider = document.querySelector<HTMLInputElement>("#time")!;
slider.max = String(Math.max(20, Number(params.get("t")) || 0));
slider.value = params.get("t") || "0.5";
const status = document.querySelector("#status")!;
try {
  if (params.has("live") || params.has("fallback")) {
    const icon = params.has("fallback") ? createFlyIcon2d(host) : createFlyIcon(host);
    if (triggers.length) icon.play(triggers, durationMs);
    slider.addEventListener("input", () => { if (triggers.length) icon.play(triggers, durationMs); });
    window.addEventListener("pagehide", () => icon.dispose(), { once: true });
  } else {
    const scene = await createFlyScene(host);
    const labels = createAnswerLabels(host);
    const activeTurn = triggers.find((trigger): trigger is TurnTrigger => trigger.kind === "turn");
    const target = triggerDrive(triggers);
    const draw = () => {
      const seconds = Number(slider.value);
      const reduced = params.has("reduced") || matchMedia("(prefers-reduced-motion: reduce)").matches;
      const duration = durationMs / 1000;
      const answerPose = answerTurnAt(activeTurn, seconds, duration, reduced);
      const end = duration + (activeTurn ? ANSWER_HOLD_SECONDS + ANSWER_RELEASE_SECONDS : 0);
      const idle = { seconds: animation === "idle" ? seconds : Math.max(0, seconds - end),
        playing: animation !== "idle" && (seconds < duration || answerPose.active),
        recovery: animation === "idle" ? 1 : seconds - end, seed };
      const pose = poseAt(reduced && seconds < duration ? target : driveAt(target, seconds / duration), seconds, reduced, idle);
      pose.yaw = answerPose.yaw;
      scene.render(pose);
      labels.render(activeTurn, answerPose.opacity, host.clientWidth, scene.projectAnswer, answerPose.chosen);
      document.querySelector("#moment")!.textContent = `${seconds.toFixed(2)} s`;
      host.dataset.time = String(seconds);
    };
    draw();
    slider.addEventListener("input", draw);
    window.addEventListener("pagehide", () => scene.dispose(), { once: true });
  }
  status.textContent = `Ready · seed ${seed} · first flurry ${flurryStart(0, seed).toFixed(3)}–${(flurryStart(0, seed) + flurryDuration(0, seed)).toFixed(3)} s`;
  document.documentElement.dataset.ready = "true";
} catch (error) {
  status.textContent = String(error);
  throw error;
}
