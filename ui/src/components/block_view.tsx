import { createSignal, Show } from "solid-js";
import { Block } from "../bindings/Block.ts";
import { Backend } from "../backend.tsx";
import { HighlightedTextarea, HighlightedTextareaRef } from "./highlighted_textarea.tsx";
import { ResultView } from "./result_view.tsx";
import { panelRef } from "./completion_panel.tsx";
import * as Icon from "./feather.tsx";

export function BlockView(props: { block: Block; backend: Backend }) {
    const [query, setQuery] = createSignal(props.block.query);
    const [submitting, setSubmitting] = createSignal(false);
    let textareaRef: HighlightedTextareaRef | null = null;

    async function submit() {
        const q = query().trim();
        if (!q || submitting()) return;
        setSubmitting(true);
        await props.backend.submitBlock(props.block.id, q);
        setSubmitting(false);
    }

    function onArrowNav(dir: "up" | "down" | "left" | "right") {
        panelRef?.navigate(dir);
    }

    function onTab() {
        if (!textareaRef || !panelRef) return;
        const completion = panelRef.currentCompletion();
        if (!completion) return;

        const cursor = textareaRef.getCursorPos();
        const text = query();

        // Walk backwards from cursor to find start of current word
        let wordStart = cursor;
        while (wordStart > 0 && !/[\s\n]/.test(text[wordStart - 1])) {
            wordStart--;
        }

        // Always replace the current partial word with the completion
        textareaRef.insertAt(wordStart, cursor, completion);
    }

    return (
        <Show
            when={props.block.state === "Composing"}
            fallback={
                <div class={`rounded bg-stone-800 ${props.block.state === "Executing" ? "opacity-50" : ""}`}>
                    <pre class="text-stone-500 text-sm font-mono px-3 pt-2 pb-2
                                whitespace-pre-wrap break-words">
                        {props.block.query || <span class="italic text-stone-700">empty</span>}
                    </pre>
                    <Show when={props.block.state === "Executing"}>
                        <div class="px-3 py-2 text-stone-500 text-sm animate-pulse">executing…</div>
                    </Show>
                    <Show when={props.block.state === "Done"}>
                        <div class="rounded-b bg-stone-900/60">
                            <ResultView result={props.block.result} backend={props.backend} />
                        </div>
                    </Show>
                </div>
            }
        >
            {/* Composing block — no ring, orange submit button top-right */}
            <div class="relative rounded bg-stone-800 px-3 pt-2 pb-2">
                <button
                    onClick={submit}
                    disabled={submitting()}
                    class="absolute top-2 right-2 rounded bg-orange-500 hover:bg-orange-400
                           w-7 h-7 flex items-center justify-center
                           transition-colors disabled:opacity-30"
                    title="Submit (Enter)"
                >
                    <Icon.Terminal size={15} stroke="#292524" stroke-width={2} />
                </button>

                <div class="pr-6">
                    <HighlightedTextarea
                        value={query()}
                        onChange={(v) => setQuery(v)}
                        prefix={props.backend.settings[0].placeholder_prefix}
                        onSubmit={submit}
                        onTab={onTab}
                        onArrowNav={onArrowNav}
                        ref={(r) => { textareaRef = r; }}
                    />
                </div>

                <div class="flex gap-4 mt-1.5 text-stone-500 text-xs select-none">
                    <span>↵ submit</span>
                    <span>⇧↵ newline</span>
                    <span>Tab complete</span>
                    <span>↑↓ navigate</span>
                </div>
            </div>
        </Show>
    );
}
