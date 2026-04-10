import { createSignal, For } from "solid-js";
import { Suggestion } from "../bindings/Suggestion.ts";
import { Backend } from "../backend.tsx";

const MAX_K = 4;

// Category tag colors
const CATEGORY_COLOR: Record<string, string> = {
    placeholder: "text-amber-500",
    keyword: "text-orange-400",
    table: "text-orange-300",
    field: "text-amber-300",
    "": "text-stone-500",
};

function categoryColor(metadata: string): string {
    for (const [key, cls] of Object.entries(CATEGORY_COLOR)) {
        if (key && metadata.toLowerCase().includes(key)) return cls;
    }
    return "text-stone-500";
}

export function CompletionPanel(props: { backend: Backend }) {
    const [selectedIndex, setSelectedIndex] = createSignal(0);

    // Merge all suggestion groups into one flat list, placeholders first
    const items = (): Suggestion[] => {
        const s = props.backend.suggestions[0];
        return [...s.placeholders, ...s.schema, ...s.other].slice(0, MAX_K);
    };

    function navigate(dir: "up" | "down" | "left" | "right") {
        const count = items().length;
        if (!count) return;
        if (dir === "down" || dir === "right") {
            setSelectedIndex((i) => Math.min(i + 1, count - 1));
        } else if (dir === "up" || dir === "left") {
            setSelectedIndex((i) => Math.max(i - 1, 0));
        }
    }

    function currentCompletion(): string | null {
        return items()[selectedIndex()]?.text ?? null;
    }

    // Module-level ref updated each render so BlockView can call navigate/currentCompletion
    panelRef = { navigate, currentCompletion };

    return (
        <div class="mt-1 overflow-hidden">
            <For each={items()}>
                {(item, i) => (
                    <div
                        class={`flex items-baseline justify-between px-3 py-1.5 gap-4
                            ${i() === selectedIndex()
                                ? "bg-stone-700"
                                : "hover:bg-stone-750"
                            }`}
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
                        <span class={`text-xs shrink-0 ${categoryColor(item.metadata)}`}>
                            {item.metadata}
                        </span>
                    </div>
                )}
            </For>
        </div>
    );
}

export let panelRef: {
    navigate: (dir: "up" | "down" | "left" | "right") => void;
    currentCompletion: () => string | null;
} | null = null;
