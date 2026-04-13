import "./App.css";
import { createSignal, For, onCleanup, onMount, Show } from "solid-js";
import * as Icon from "./components/feather.tsx";
import { BlockView } from "./components/block_view.tsx";
import { CompletionPanel } from "./components/completion_panel.tsx";
import { Sidebar } from "./components/sidebar.tsx";
import { Backend } from "./backend.tsx";
import { jumpToComposing } from "./composing.ts";

type SidebarTab = "values" | "settings" | "nav" | "tasks" | null;

export default function App() {
    const backend = new Backend();
    const [sidebarTab, setSidebarTab] = createSignal<SidebarTab>(null);

    onMount(async () => {
        await backend.init();
        requestAnimationFrame(() => jumpToComposing());

        function onKeyDown(e: KeyboardEvent) {
            if (e.ctrlKey && e.key === "/") {
                e.preventDefault();
                jumpToComposing();
            }
        }
        document.addEventListener("keydown", onKeyDown);
        onCleanup(() => document.removeEventListener("keydown", onKeyDown));
    });

    return (
        <main class="h-screen w-screen bg-stone-900 flex overflow-hidden text-stone-200">
            {/* Single scroll context */}
            <div id="scroll-root" class="flex-1 overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden min-w-0">
                <div class="pl-3 pr-0 pt-3 pb-8 space-y-2">
                    <For each={backend.blocks[0]}>
                        {(block) => <BlockView block={block} backend={backend} />}
                    </For>
                    <Show when={backend.composingBlock()}>
                        <CompletionPanel backend={backend} />
                    </Show>
                </div>
            </div>

            {/* Sidebar panel */}
            <Show when={sidebarTab()}>
                <Sidebar
                    tab={sidebarTab()!}
                    backend={backend}
                    onClose={() => setSidebarTab(null)}
                />
            </Show>

            {/* Right button column — values top, TBD mid, settings bottom */}
            <div class="w-12 flex flex-col items-center gap-3 py-3 shrink-0">
                <button
                    onClick={() => setSidebarTab(sidebarTab() === "values" ? null : "values")}
                    class={`rounded-md w-8 h-8 flex items-center justify-center transition-colors
                        ${sidebarTab() === "values"
                            ? "bg-amber-500/20 text-amber-400"
                            : "bg-stone-800 text-stone-500 hover:text-stone-300"}`}
                    title="Values"
                >
                    <Icon.Paperclip size={15} stroke="currentColor" />
                </button>
                <button
                    onClick={() => setSidebarTab(sidebarTab() === "nav" ? null : "nav")}
                    class={`rounded-md w-8 h-8 flex items-center justify-center transition-colors
                        ${sidebarTab() === "nav"
                            ? "bg-stone-700/50 text-stone-200"
                            : "bg-stone-800 text-stone-500 hover:text-stone-300"}`}
                    title="Query history"
                >
                    <Icon.List size={15} stroke="currentColor" />
                </button>
                <button
                    onClick={() => setSidebarTab(sidebarTab() === "tasks" ? null : "tasks")}
                    class={`rounded-md w-8 h-8 flex items-center justify-center transition-colors
                        ${sidebarTab() === "tasks"
                            ? "bg-lime-500/20 text-lime-400"
                            : "bg-stone-800 text-stone-500 hover:text-stone-300"}`}
                    title="Tasks"
                >
                    <Icon.Play size={13} stroke="currentColor" />
                </button>
                {/* Settings at bottom */}
                <div class="flex-1 flex items-end pb-1">
                    <button
                        onClick={() => setSidebarTab(sidebarTab() === "settings" ? null : "settings")}
                        class={`rounded-md w-8 h-8 flex items-center justify-center transition-colors
                            ${sidebarTab() === "settings"
                                ? "bg-stone-700 text-stone-200"
                                : "bg-stone-800 text-stone-500 hover:text-stone-300"}`}
                        title="Settings"
                    >
                        <Icon.Settings size={15} stroke="currentColor" />
                    </button>
                </div>
            </div>
        </main>
    );
}
