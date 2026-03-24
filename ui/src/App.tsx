import "./App.css";
import { QueryBar } from "./components/QueryBar.tsx";
import { Settings } from "./components/Settings.tsx";
import * as Icon from "./components/Feather.tsx";
import {createSignal, For} from "solid-js";
import { animate } from "motion";
import {Block, BlockData} from "./components/Block.tsx";
import {createStore} from "solid-js/store";

export default function App() {
  const [open, setOpen] = createSignal(false);
  const [blocks, set_blocks] = createStore<BlockData[]>([]);
  const [prompt, set_prompt] = createSignal("");
  let containerRef!: HTMLDivElement;
  let inputRef!: HTMLDivElement;
  let settingsRef!: HTMLDivElement;
  let gearRef!: HTMLDivElement;
  let chevRef!: HTMLDivElement;
  let settingsBtnRef!: HTMLButtonElement;

  let currentAnim: ReturnType<typeof animate> | null = null;
  let closedHeight = 0;

  function toggle() {
    if (currentAnim) {
      currentAnim.stop();
      currentAnim = null;
    }

    const willOpen = !open();
    setOpen(willOpen);

    // Instant button swap
    settingsBtnRef.style.backgroundColor = willOpen ? "#ffffff" : "#404040";
    (gearRef as HTMLElement).style.opacity = willOpen ? "0" : "1";
    (chevRef as HTMLElement).style.opacity = willOpen ? "1" : "0";

    if (willOpen) {
      closedHeight = containerRef.offsetHeight;
      const targetHeight = window.innerHeight - 32;

      currentAnim = animate([
        // 1. Fade out input (button swaps instantly below)
        [inputRef, { opacity: 0 }, { duration: 0.25 }],
        // 2. Expand container (explicit from→to keyframes)
        [containerRef, { height: [`${closedHeight}px`, `${targetHeight}px`] }, { duration: 0.4, ease: [0.4, 0, 0.2, 1] }],
        // 3. Fade in settings
        [settingsRef, { opacity: 1 }, { duration: 0.3 }],
      ]);

      // After animation lands, swap to responsive unit
      currentAnim.then(() => {
        containerRef.style.height = "calc(100vh - 32px)";
      }).catch(() => {});
    } else {
      currentAnim = animate([
        // 1. Fade out settings
        [settingsRef, { opacity: 0 }, { duration: 0.25 }],
        // 2. Collapse container
        [containerRef, { height: `${closedHeight}px` }, { duration: 0.4, ease: [0.4, 0, 0.2, 1] }],
        // 3. Fade in input (button swaps instantly below)
        [inputRef, { opacity: 1 }, { duration: 0.25 }],
      ]);

      // Clean up explicit height after close finishes
      currentAnim.then(() => {
        containerRef.style.height = "";
      }).catch(() => {});
    }
  }
  function execute() {
    const block: BlockData = {
      query: prompt(),
    };
    console.log("appending", block);
    set_blocks([...blocks, block]);
    set_prompt("");
  }
  return (
      <main class="relative h-screen w-screen bg-stone-800 flex flex-col overflow-hidden">
        <div
            class={"absolute right-2 top-2 rounded-md bg-stone-600 h-10 w-10 z-10 items-center justify-center flex"}>
          <Icon.Settings size={16} stroke={"#d4d4d4"}/>
        </div>
        {/* Canvas — output blocks */}
        <div class="relative flex-1 overflow-y-scroll">
          <div class={" w-full min-h-full flex flex-col gap-4 p-4"}>
            <For each={blocks} fallback={<div>{"nothing"}</div>}>{(bd) => {
              return <Block data={bd}/>;
            }}</For>
          </div>
        </div>

        {/* Query row */}
        <div class="flex items-end gap-3 p-4">
          <div
              ref={containerRef!}
              class="relative w-full overflow-hidden rounded-2xl bg-stone-700"
          >
            {/* Input */}
            <div ref={inputRef!}>
              <QueryBar input={prompt} set_input={set_prompt}/>
            </div>
            {/* Settings overlay */}
            <div
                ref={settingsRef!}
                class="absolute inset-0 overflow-auto"
                style={{opacity: "0", "pointer-events": open() ? "auto" : "none"}}
            >
              <Settings/>
            </div>
          </div>

          <div class="flex flex-col gap-3 shrink-0">
            <button
                ref={settingsBtnRef!}
                onClick={toggle}
                class="relative w-10 h-10 rounded-full bg-neutral-700 flex items-center justify-center focus:outline-none cursor-pointer"
            >
              <div ref={gearRef!} class="absolute">
                <Icon.Settings size={16} stroke="#d4d4d4"/>
              </div>
              <div ref={chevRef!} class="absolute" style={{opacity: "0"}}>
                <Icon.ChevronDown size={16} stroke="#1a1a1a" strokeWidth={2.2}/>
              </div>
            </button>
            <button
                onKeyDown={(e) => {
                  if (e.key == "Enter" && e.shiftKey) {
                    execute()
                  }
                }}
                onClick={(_) => {
                  execute()
                }}
                class="w-10 h-10 rounded-full flex items-center justify-center focus:outline-none bg-orange-300"
            >
              <Icon.Terminal size={17} stroke="#1a1a1a" strokeWidth={2.2}/>
            </button>
          </div>
        </div>

      </main>
  );
}
