import { createSignal, JSX, Show } from "solid-js";
import { Block } from "../bindings/Block.ts";
import { Backend } from "../backend.tsx";
import { HighlightedTextarea, HighlightedTextareaRef } from "./highlighted_textarea.tsx";
import { ResultView } from "./result_view.tsx";
import { panelRef } from "./completion_panel.tsx";
import { setComposingEls } from "../composing.ts";
import * as Icon from "./feather.tsx";

function Kbd(props: { children: JSX.Element }) {
    return (
        <kbd class="inline-flex items-center gap-0.5 px-1 py-1 rounded
                    bg-stone-900 text-stone-500 border border-stone-700 border-b-2
                    leading-none select-none">
            {props.children}
        </kbd>
    );
}

export function BlockView(props: { block: Block; backend: Backend }) {
    const [query, setQuery] = createSignal(props.block.query);
    const [submitting, setSubmitting] = createSignal(false);
    let textareaRef: HighlightedTextareaRef | null = null;

    // History walk: index into previous blocks when ↑ pressed in empty textarea
    let historyIdx = -1;

    async function submit() {
        const q = query().trim();
        if (!q || submitting()) return;
        historyIdx = -1;
        setSubmitting(true);
        await props.backend.submitBlock(props.block.id, q);
        setSubmitting(false);
    }

    function onArrowNav(dir: "up" | "down" | "left" | "right") {
        panelRef?.navigate(dir);
    }

    async function onPaste(text: string): Promise<string> {
        const name = await props.backend.pasteValue(text.slice(0, 48), text);
        return `${props.backend.settings[0].placeholder_prefix}${name}`;
    }

    function onHistory(dir: "up" | "down") {
        const done = props.backend.blocks[0]
            .filter((b) => b.state === "Done" && b.query.trim())
            .map((b) => b.query);
        if (!done.length) return;

        if (dir === "up") {
            historyIdx = Math.min(historyIdx + 1, done.length - 1);
        } else {
            historyIdx = Math.max(historyIdx - 1, -1);
        }

        setQuery(historyIdx === -1 ? "" : done[done.length - 1 - historyIdx]);
    }

    function onTab() {
        if (!textareaRef || !panelRef) return;
        const completion = panelRef.currentCompletion();
        if (!completion) return;

        const cursor = textareaRef.getCursorPos();
        const text = query();

        let wordStart = cursor;
        while (wordStart > 0 && !/[\s\n]/.test(text[wordStart - 1])) {
            wordStart--;
        }
        textareaRef.insertAt(wordStart, cursor, completion);
    }

    return (
        <Show
            when={props.block.state === "Composing"}
            fallback={
                <div id={`block-${props.block.id}`} class={`rounded bg-stone-800 ${props.block.state === "Executing" ? "opacity-50" : ""}`}>
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
            {/* Composing block */}
            <div
                ref={(el) => {
                    requestAnimationFrame(() => {
                        const ta = el.querySelector("textarea");
                        if (ta) setComposingEls(ta as HTMLTextAreaElement, el);
                    });
                }}
                class="relative rounded bg-stone-800 px-3 pt-2 pb-2"
            >
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
                        onChange={(v) => {
                            setQuery(v);
                            historyIdx = -1;
                            requestAnimationFrame(() => {
                                const scroller = document.getElementById("scroll-root");
                                if (scroller) scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
                            });
                        }}
                        prefix={props.backend.settings[0].placeholder_prefix}
                        onSubmit={submit}
                        onTab={onTab}
                        onArrowNav={onArrowNav}
                        onHistory={onHistory}
                        onPaste={onPaste}
                        ref={(r) => { textareaRef = r; }}
                    />
                </div>

                <div class="flex flex-wrap gap-x-3 gap-y-1 mt-1.5 text-stone-400 text-xs select-none items-center">
                    <span class="inline-flex items-center gap-1"><Kbd><Icon.CornerDownLeft size={11} /></Kbd> submit</span>
                    <span class="inline-flex items-center gap-1"><Kbd><Icon.ShiftKey size={11} /><Icon.CornerDownLeft size={11} /></Kbd> newline</span>
                    <span class="inline-flex items-center gap-1"><Kbd><Icon.TabKey size={11} /></Kbd> complete</span>
                    <span class="inline-flex items-center gap-1"><Kbd><Icon.ArrowUp size={11} /><Icon.ArrowDown size={11} /></Kbd> navigate</span>
                    <span class="inline-flex items-center gap-1"><Kbd><Icon.Option size={11} /><Icon.ArrowUp size={11} /><Icon.ArrowDown size={11} /></Kbd> history</span>
                    <span class="inline-flex items-center gap-1"><Kbd><span class="text-xs font-mono">ctrl</span><span class="text-xs font-mono">/</span></Kbd> focus</span>
                </div>
            </div>
        </Show>
    );
}
