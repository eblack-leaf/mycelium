import { For } from "solid-js";
import { Suggestion } from "../bindings/Suggestion.ts";
import { Backend } from "../backend.tsx";

const MAX_K = 4;

function categoryColor(metadata: string): string {
    if (metadata.includes("placeholder")) return "text-amber-500";
    if (metadata.includes("keyword")) return "text-orange-400";
    if (metadata.includes("table")) return "text-orange-300";
    if (metadata.includes("field")) return "text-amber-300";
    return "text-stone-500";
}

export function CompletionPanel(props: { backend: Backend }) {
    const items = (): Suggestion[] => {
        const s = props.backend.suggestions[0];
        return [...s.placeholders, ...s.schema, ...s.other].slice(0, MAX_K);
    };

    panelRef = {
        currentCompletion: () => items()[0]?.text ?? null,
        getItemAt: (i: number) => items()[i]?.text ?? null,
        resetIndex: () => {},
    };

    return (
        <div class="mt-1 overflow-hidden" style={{ "min-height": `${MAX_K * 2}rem` }}>
            <For each={items()}>
                {(item, i) => (
                    <div
                        onClick={() => insertCompletionFn?.(item.text)}
                        class="flex items-center justify-between px-3 py-1.5 gap-4 cursor-pointer hover:bg-stone-700/50"
                    >
                        <span class={`font-mono text-sm ${
                            item.metadata === "placeholder"
                                ? "text-amber-400"
                                : item.metadata === "keyword"
                                ? "text-orange-400"
                                : "text-stone-300"
                        }`}>
                            {item.text}
                        </span>
                        <span class="flex items-center gap-1.5 shrink-0">
                            <span class={`text-xs ${categoryColor(item.metadata)}`}>
                                {item.metadata}
                            </span>
                            {i() === 0 && (
                                <kbd class="px-1 py-0.5 rounded text-xs bg-stone-900 text-orange-400 border border-stone-700 border-b-2 leading-none select-none">
                                    tab
                                </kbd>
                            )}
                            <kbd class="px-1 py-0.5 rounded text-xs bg-stone-900 text-stone-500 border border-stone-700 border-b-2 leading-none select-none">
                                {i() + 1}
                            </kbd>
                        </span>
                    </div>
                )}
            </For>
        </div>
    );
}

export let panelRef: {
    currentCompletion: () => string | null;
    getItemAt: (i: number) => string | null;
    resetIndex: () => void;
} | null = null;

// Registered by the active BlockView
export let insertCompletionFn: ((text: string) => void) | null = null;
export function registerInsertCompletion(fn: (text: string) => void) {
    insertCompletionFn = fn;
}
