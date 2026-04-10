import { createSignal, For, JSX, Show } from "solid-js";
import { Backend } from "../backend.tsx";
import * as Icon from "./feather.tsx";

function typeMeta(value: unknown): string {
    if (value === null) return "null";
    if (typeof value === "boolean") return "bool";
    if (typeof value === "number") return "number";
    if (typeof value === "string") return "string";
    if (Array.isArray(value)) return `[${(value as unknown[]).length}]`;
    if (typeof value === "object") return `{${Object.keys(value as object).length}}`;
    return typeof value;
}

function valueDisplay(value: unknown): JSX.Element {
    if (value === null) return <span class="text-stone-500">null</span>;
    if (typeof value === "boolean") return <span class="text-orange-400">{String(value)}</span>;
    if (typeof value === "number") return <span class="text-amber-300">{String(value)}</span>;
    if (typeof value === "string") return <span class="text-amber-400/90">"{value}"</span>;
    if (Array.isArray(value)) return <span class="text-stone-500">[…]</span>;
    if (typeof value === "object") return <span class="text-stone-500">{"{"}&hellip;{"}"}</span>;
    return <span class="text-stone-400">{String(value)}</span>;
}

interface RowProps {
    label: string | null;
    value: unknown;
    depth: number;
    backend: Backend;
}

function JsonRow(props: RowProps): JSX.Element {
    const [saving, setSaving] = createSignal(false);
    const [name, setName] = createSignal("");
    const [loading, setLoading] = createSignal(false);
    async function openSave() {
        if (saving()) { setSaving(false); return; }
        setLoading(true);
        setSaving(true);
        const suggested = await props.backend.suggestName(
            JSON.stringify(props.value).slice(0, 48)
        );
        setName(suggested);
        setLoading(false);
    }

    async function confirm() {
        await props.backend.saveValue(name(), JSON.stringify(props.value));
        setSaving(false);
    }

    const indent = () => props.depth * 14;

    return (
        <>
            <div
                class="flex items-center gap-2 h-8 group"
                style={{ "padding-left": `${indent()}px` }}
            >
                {/* Key */}
                <Show when={props.label !== null}>
                    <span class="text-stone-400 font-mono text-sm shrink-0">"{props.label}":</span>
                </Show>

                {/* Value */}
                <span class="font-mono text-sm shrink-0">
                    {valueDisplay(props.value)}
                </span>

                {/* Type annotation — right after value */}
                <span class="text-stone-600 text-xs shrink-0">{typeMeta(props.value)}</span>

                {/* Save trigger — right after annotation */}
                <button
                    onClick={openSave}
                    onKeyDown={(e) => { if (e.key === "Escape") setSaving(false); }}
                    class={`shrink-0 flex items-center rounded-sm px-1 py-0.5 transition-colors
                        ${saving()
                            ? "text-amber-400 bg-stone-700"
                            : "text-stone-700 hover:text-amber-500 hover:bg-stone-700 group-hover:text-stone-500"
                        }`}
                >
                    <Icon.ChevronsRight size={18} stroke="currentColor" stroke-width={2} />
                </button>

                {/* Inline save form — appears to the right of >> */}
                <Show when={saving()}>
                    <Show when={loading()} fallback={
                        <span class="inline-flex items-center gap-2 shrink-0">
                            <input
                                class="bg-stone-700 text-stone-100 text-sm font-mono
                                       rounded px-2.5 outline-none w-32 h-6"
                                value={name()}
                                onInput={(e) => setName(e.currentTarget.value)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") confirm();
                                    if (e.key === "Escape") setSaving(false);
                                }}
                                onBlur={(e) => {
                                    const related = e.relatedTarget as HTMLElement | null;
                                    if (!related || !related.closest("[data-save-row]")) {
                                        setTimeout(() => setSaving(false), 120);
                                    }
                                }}
                                autofocus
                            />
                            <button
                                data-save-row
                                onClick={confirm}
                                class="text-sm text-amber-400 hover:text-amber-300 transition-colors shrink-0"
                            >
                                save
                            </button>
                        </span>
                    }>
                        <span class="text-stone-600 text-sm italic shrink-0">…</span>
                    </Show>
                </Show>
            </div>

            {/* Recurse into children — no collapse, always shown */}
            <Show when={Array.isArray(props.value)}>
                <For each={props.value as unknown[]}>
                    {(item, i) => (
                        <JsonRow
                            label={String(i())}
                            value={item}
                            depth={props.depth + 1}
                            backend={props.backend}
                        />
                    )}
                </For>
            </Show>
            <Show when={!Array.isArray(props.value) && typeof props.value === "object" && props.value !== null}>
                <For each={Object.entries(props.value as Record<string, unknown>)}>
                    {([key, val]) => (
                        <JsonRow
                            label={key}
                            value={val}
                            depth={props.depth + 1}
                            backend={props.backend}
                        />
                    )}
                </For>
            </Show>
        </>
    );
}

export function ResultView(props: { result: string | null; backend: Backend }) {
    if (!props.result) return null;

    let parsed: unknown;
    try {
        parsed = JSON.parse(props.result);
    } catch {
        return <pre class="text-red-400 text-sm font-mono px-3 py-2">{props.result}</pre>;
    }

    return (
        <div class="px-3 py-2">
            <Show when={Array.isArray(parsed)}>
                <For each={parsed as unknown[]}>
                    {(item, i) => (
                        <JsonRow
                            label={String(i())}
                            value={item}
                            depth={0}
                            backend={props.backend}
                        />
                    )}
                </For>
            </Show>
            <Show when={!Array.isArray(parsed)}>
                <JsonRow label={null} value={parsed} depth={0} backend={props.backend} />
            </Show>
        </div>
    );
}
