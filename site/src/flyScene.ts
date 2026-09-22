import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { answerYaw, HOVER_HEIGHT, solveLeg, type FlyPose, type SolvedLeg, type Vec3 } from "./flyPose";
import type { FlySide } from "./states";
import type { LabelPoint } from "./flyAnswer";

export interface FlyScene {
  render(pose: FlyPose): void;
  resize(size: number): void;
  projectAnswer(side: FlySide): LabelPoint;
  dispose(): void;
}

export function createFlyCamera(): THREE.OrthographicCamera {
  const camera = new THREE.OrthographicCamera(-3.3, 3.3, 3.3, -3.3, .1, 40);
  camera.position.set(4.1, 8.4, -6.2);
  camera.lookAt(0, -.1, .55);
  camera.updateMatrixWorld(true);
  return camera;
}

/** Project beyond the anatomical head along local -Z at the final heading.
 * The camera sees physical left on screen right; DOM left/right would invert the answer. */
export function projectAnswerPoint(camera: THREE.Camera, head: THREE.Vector3, side: FlySide): LabelPoint {
  const axis = new THREE.Vector3(0, 1, 0);
  const start = head.clone().applyAxisAngle(axis, answerYaw(side)).project(camera);
  const point = head.clone();
  point.z -= 3;
  point.applyAxisAngle(axis, answerYaw(side)).project(camera);
  // Fit along the projected forward ray, preserving its direction while reserving text space.
  let distance = 1;
  for (const component of ["x", "y"] as const) {
    const delta = point[component] - start[component];
    if (delta !== 0) distance = Math.min(distance, (Math.sign(delta) * .72 - start[component]) / delta);
  }
  point.lerpVectors(start, point, distance);
  return { x: (point.x + 1) / 2, y: (1 - point.y) / 2 };
}

function releaseResources(root: THREE.Object3D) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  root.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    geometries.add(object.geometry);
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      materials.add(material);
      for (const value of Object.values(material)) if (value instanceof THREE.Texture) textures.add(value);
    }
  });
  geometries.forEach(geometry => geometry.dispose());
  materials.forEach(material => material.dispose());
  textures.forEach(texture => texture.dispose());
}

// Geometry and joint pivots: flybody, Apache-2.0; see public/models/fly-NOTICE.txt.
// Materials remove biological colour so only the answer indicator uses cyan.
export async function createFlyScene(host: HTMLElement, onContextLost = () => {}, signal?: AbortSignal): Promise<FlyScene> {
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(0xffffff, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  const canvas = renderer.domElement;
  canvas.style.cssText = "display:block;max-width:100%;filter:none";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Three-dimensional fruit fly specimen, showing its triggered behavioural response");
  const contextLost = (event: Event) => { event.preventDefault(); onContextLost(); };
  canvas.addEventListener("webglcontextlost", contextLost);
  host.appendChild(canvas);
  const scene = new THREE.Scene();
  const camera = createFlyCamera();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xaca49a, 1.15));
  const key = new THREE.DirectionalLight(0xffffff, 2);
  key.position.set(-3, 7, -4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xffffff, 1.6);
  rim.position.set(3, 2, 5);
  scene.add(rim);

  // An analytic contact shadow avoids a shadow-map pass on mobile and stays soft at 200px.
  // It stays on the ground, so a hop separates the specimen from its shadow.
  const shadowMaterial = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { opacity: { value: .2 } },
    vertexShader: "varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}",
    fragmentShader: "varying vec2 vUv; uniform float opacity; void main(){vec2 p=(vUv-.5)*2.;float a=exp(-dot(p,p)*4.)*(1.-smoothstep(.5,1.,length(p)));gl_FragColor=vec4(.15,.13,.11,a*opacity);}",
  });
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(2.9, 3.6), shadowMaterial);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.set(0, -1.33, .3);
  scene.add(shadow);
  const fly = new THREE.Group();
  scene.add(fly);

  let disposed = false;
  function dispose() {
    if (disposed) return;
    disposed = true;
    canvas.removeEventListener("webglcontextlost", contextLost);
    signal?.removeEventListener("abort", dispose);
    releaseResources(scene);
    renderer.dispose();
    renderer.forceContextLoss();
    canvas.remove();
  }
  signal?.addEventListener("abort", dispose, { once: true });

  try {
    const response = await fetch(`${import.meta.env.BASE_URL}models/fly.glb`, { signal });
    if (!response.ok) throw new Error(`Fly model: HTTP ${response.status}`);
    const gltf = await new GLTFLoader().parseAsync(await response.arrayBuffer(), "");
    if (disposed || signal?.aborted) {
      releaseResources(gltf.scene);
      throw new DOMException("Fly disposed during loading", "AbortError");
    }
    fly.add(gltf.scene);
    gltf.scene.traverse(object => {
      if (!(object instanceof THREE.Mesh)) return;
      const material = object.material as THREE.MeshStandardMaterial;
      if (material.name === "red") material.flatShading = true;
      if (material.name === "membrane") {
        material.depthWrite = false;
        object.renderOrder = 2;
      }
    });
    const node = (name: string) => {
      const result = gltf.scene.getObjectByName(name);
      if (!result) throw new Error(`Missing fly joint: ${name}`);
      return result;
    };
    const wings = [node("wing_left"), node("wing_right")];
    fly.updateMatrixWorld(true);
    const head = fly.worldToLocal(node("head").getWorldPosition(new THREE.Vector3()));
    const answerPoints = {
      left: projectAnswerPoint(camera, head, "left"),
      right: projectAnswerPoint(camera, head, "right"),
    };
    // Two translucent exposure samples per wing keep the buzz legible even in a
    // frozen frame (and avoid frame-rate aliasing). Geometry is shared, created once.
    const blurMaterial = new THREE.MeshBasicMaterial({ color: 0xb8b3a8,
      transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide });
    const wingGhosts = wings.map(wing => [-1, 1].map(() => {
      const ghost = wing.clone(true);
      ghost.traverse(object => {
        if (object instanceof THREE.Mesh) { object.material = blurMaterial; object.renderOrder = 1; }
      });
      wing.parent!.add(ghost);
      return ghost;
    }));
    const antennae = [node("antenna_left"), node("antenna_right")];
    const proboscis = node("proboscis");
    const haustellum = node("haustellum");
    const legs = ["left", "right"].flatMap(side => [1, 2, 3].map(pair => {
      const femur = node(`femur_T${pair}_${side}`), tibia = node(`tibia_T${pair}_${side}`);
      const tarsus = node(`tarsus_T${pair}_${side}`);
      return { femur, tibia, tarsus, upper: tibia.position.toArray() as Vec3,
        lower: tarsus.position.toArray() as Vec3,
        upperAxis: tibia.position.clone().normalize(), lowerAxis: tarsus.position.clone().normalize() };
    }));
    const knee = new THREE.Vector3(), ankle = new THREE.Vector3();
    const lowerRotation = new THREE.Quaternion();

    const offset: Vec3 = [0, 0, 0];
    const solved: SolvedLeg = { knee: [0, 0, 0], ankle: [0, 0, 0] };

    function render(pose: FlyPose) {
      if (disposed) return;
      fly.rotation.y = pose.yaw;
      fly.position.set(pose.sway, pose.height, pose.backward);
      for (let i = 0; i < wings.length; i++) {
        const wing = wings[i];
        const side = i === 0 ? 1 : -1;
        wing.rotation.set(0, side * (1.5 - pose.wings[i]), -side * pose.wingLift[i]);
        for (let sample = 0; sample < 2; sample++) {
          const ghost = wingGhosts[i][sample];
          const spread = sample === 0 ? -1 : 1;
          ghost.visible = pose.wingBlur > .001;
          ghost.rotation.set(0, side * (1.5 - pose.wings[i] + spread * .11 * pose.wingBlur),
            -side * (pose.wingLift[i] + spread * .19 * pose.wingBlur));
        }
      }
      blurMaterial.opacity = pose.wingBlur * .105;
      for (let i = 0; i < antennae.length; i++) antennae[i].rotation.y = pose.antennae[i];
      proboscis.rotation.x = -.95 + pose.proboscis * 3.15;
      haustellum.rotation.x = -pose.proboscis * 1.6;
      for (let i = 0; i < legs.length; i++) {
        const leg = legs[i];
        // Feet remain level during stance; crouching lowers the body between the bent legs.
        offset[0] = pose.legs[i].inward;
        offset[1] = pose.legs[i].lift - Math.min(0, pose.height);
        offset[2] = pose.legs[i].sweep;
        solveLeg(leg.upper, leg.lower, offset, solved);
        knee.fromArray(solved.knee);
        ankle.fromArray(solved.ankle).sub(knee).normalize();
        leg.femur.quaternion.setFromUnitVectors(leg.upperAxis, knee.normalize());
        lowerRotation.setFromUnitVectors(leg.lowerAxis, ankle);
        leg.tibia.quaternion.copy(leg.femur.quaternion).invert().multiply(lowerRotation);
        leg.tarsus.quaternion.copy(lowerRotation).invert();
        leg.tarsus.rotateZ(pose.legs[i].curl);
      }
      shadow.position.z = .3 + pose.backward;
      shadow.rotation.z = pose.yaw;
      const separation = Math.min(1, Math.max(0, pose.height) / HOVER_HEIGHT);
      shadow.scale.setScalar(1 - .3 * separation);
      shadowMaterial.uniforms.opacity.value = .2 * (1 - .65 * separation);
      renderer.render(scene, camera);
    }
    function resize(size: number) {
      renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
      renderer.setSize(size, size);
    }
    resize(Math.min(host.clientWidth, host.clientHeight) || 320);
    return { render, resize, dispose, projectAnswer: side => answerPoints[side] };
  } catch (error) {
    dispose();
    throw error;
  }
}
