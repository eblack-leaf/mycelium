import { animate } from "motion";
import { createSignal, For, JSX, onMount, Show } from "solid-js";
import { Backend } from "../backend.tsx";
import * as Icon from "./feather.tsx";
import { focusComposing } from "../composing.ts";

// Flatten JSON into ordered rows — parent rows first, then children
interface FlatRow { depth: number; label: string | null; value: unknown }

function flatten(value: unknown, depth: number, label: string | null, out: FlatRow[]) {
    out.push({ depth, label, value });
    if (Array.isArray(value)) {
        (value as unknown[]).forEach((v, i) => flatten(v, depth + 1, String(i), out));
    } else if (typeof value === "object" && value !== null) {
        Object.entries(value as Record<string, unknown>).forEach(([k, v]) =>
            flatten(v, depth + 1, k, out)
        );
    }
}

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
    if (Array.isArray(value)) return <span class="text-stone-500">[{(value as unknown[]).length}]</span>;
    if (typeof value === "object") return <span class="text-stone-500">{"{"}{Object.keys(value as object).length}{"}"}</span>;
    return <span class="text-stone-400">{String(value)}</span>;
}

// Single flat row — no recursion into children
function FlatJsonRow(props: {
    row: FlatRow;
    rowClass: string;
    backend: Backend;
}) {
    const [saving, setSaving] = createSignal(false);
    const [name, setName] = createSignal("");
    const [loading, setLoading] = createSignal(false);

    async function openSave() {
        if (saving()) { setSaving(false); return; }
        setLoading(true);
        setSaving(true);
        const suggested = await props.backend.suggestName(
            JSON.stringify(props.row.value).slice(0, 48)
        );
        setName(suggested);
        setLoading(false);
    }

    async function confirm() {
        await props.backend.saveValue(name(), JSON.stringify(props.row.value));
        setSaving(false);
    }

    return (
        <div
            class={`${props.rowClass} flex items-center gap-2 h-8 group`}
            style={{ opacity: 0, "padding-left": `${props.row.depth * 14}px` }}
        >
            <Show when={props.row.label !== null}>
                <span class="text-stone-400 font-mono text-sm shrink-0">"{props.row.label}":</span>
            </Show>

            <span class="font-mono text-sm shrink-0">
                {valueDisplay(props.row.value)}
            </span>

            <span class="text-stone-600 text-xs shrink-0">{typeMeta(props.row.value)}</span>

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

    const rows: FlatRow[] = [];
    if (Array.isArray(parsed)) {
        (parsed as unknown[]).forEach((v, i) => flatten(v, 0, String(i), rows));
    } else {
        flatten(parsed, 0, null, rows);
    }

    // Unique class per instance so the string selector is scoped to this result
    const cls = `jr-${Math.random().toString(36).slice(2, 7)}`;

    onMount(() => {
        const scroller = document.getElementById("scroll-root");
        const rowEls = Array.from(document.querySelectorAll(`.${cls}`)) as HTMLElement[];

        // Build a sequence: each row fades in at its stagger offset,
        // and a callback fires at that same time to scroll the row into view.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const seq: any[] = [];
        rowEls.forEach((el, i) => {
            const t = i * 0.02;
            seq.push([el, { opacity: 1 }, { duration: 0.07, at: t }]);
            seq.push([() => {
                if (!scroller) return;
                const rect = el.getBoundingClientRect();
                const scrollerRect = scroller.getBoundingClientRect();
                if (rect.bottom > scrollerRect.bottom) {
                    scroller.scrollTop += rect.bottom - scrollerRect.bottom + 8;
                }
            }, { at: t + 0.05 }]);
        });
        seq.push([() => focusComposing()]);

        animate(seq);
    });

    return (
        <div class="px-3 py-2">
            <For each={rows}>
                {(row) => (
                    <FlatJsonRow
                        row={row}
                        rowClass={cls}
                        backend={props.backend}
                    />
                )}
            </For>
        </div>
    );
}
