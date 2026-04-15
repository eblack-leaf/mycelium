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

// Serialize a value for saving as a placeholder.
// Record IDs (table:identifier) are stored raw so they work in SurrealQL directly.
// All other strings get JSON-stringified so they arrive quoted as string literals.
function serializeForSave(value: unknown): string {
    if (typeof value === "string" && /^[A-Za-z_][A-Za-z0-9_]*:[^\s"',]+$/.test(value)) {
        return value; // record ID — keep raw
    }
    return JSON.stringify(value);
}

// Single flat row — no recursion into children
function copyValue(value: unknown): string {
    // Copy raw string without quotes; everything else as JSON
    return typeof value === "string" ? value : JSON.stringify(value);
}

// Union of top-level keys across all object items, preserving first-seen order.
// Returns [] for scalar/mixed arrays with no object items.
function extractSchemaKeys(items: unknown[]): string[] {
    const seen = new Map<string, number>();
    let order = 0;
    for (const item of items) {
        if (typeof item === "object" && item !== null && !Array.isArray(item)) {
            for (const k of Object.keys(item as Record<string, unknown>)) {
                if (!seen.has(k)) seen.set(k, order++);
            }
        }
    }
    return [...seen.entries()].sort((a, b) => a[1] - b[1]).map(e => e[0]);
}

// Build a short preview string for a collapsed record.
function summarizeRecord(item: unknown): string {
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
        return String(item);
    }
    const obj = item as Record<string, unknown>;
    const parts: string[] = [];

    function shortVal(v: unknown): string {
        const s = typeof v === "string" ? v : JSON.stringify(v);
        return s.length > 22 ? s.slice(0, 20) + "…" : s;
    }

    const idKey = "id" in obj ? "id" : "_id" in obj ? "_id" : null;
    if (idKey) {
        parts.push(`${idKey}: ${shortVal(obj[idKey])}`);
    }

    let count = 0;
    const limit = idKey ? 2 : 3;
    for (const [k, v] of Object.entries(obj)) {
        if (k === idKey) continue;
        if (v !== null && typeof v === "object") continue;
        parts.push(`${k}: ${shortVal(v)}`);
        if (++count >= limit) break;
    }

    return parts.length > 0 ? parts.join(" · ") : "(no preview)";
}

function FlatJsonRow(props: {
    row: FlatRow;
    rowClass: string;
    backend: Backend;
    visible?: boolean;
}) {
    const [saving, setSaving] = createSignal(false);
    const [name, setName] = createSignal("");
    const [loading, setLoading] = createSignal(false);
    const [copied, setCopied] = createSignal(false);
    let inputEl!: HTMLInputElement;

    function copy() {
        navigator.clipboard.writeText(copyValue(props.row.value));
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
    }

    async function openSave() {
        if (saving()) { setSaving(false); return; }
        setLoading(true);
        setSaving(true);
        const suggested = await props.backend.suggestName(
            JSON.stringify(props.row.value).slice(0, 48)
        );
        setName(suggested);
        setLoading(false);
        requestAnimationFrame(() => { inputEl?.select(); });
    }

    async function confirm() {
        await props.backend.saveValue(name(), serializeForSave(props.row.value));
        setSaving(false);
    }

    return (
        <div
            class={`${props.rowClass} flex items-center gap-2 h-6 group overflow-hidden`}
            style={{ opacity: props.visible ? 1 : 0, "padding-left": `${props.row.depth * 14}px` }}
        >
            <Show when={props.row.label !== null}>
                <span class="text-stone-400 font-mono text-xs shrink-0">"{props.row.label}":</span>
            </Show>

            <span
                class="font-mono text-xs min-w-0 truncate"
                title={typeof props.row.value === "string"
                    ? props.row.value
                    : JSON.stringify(props.row.value)}
            >
                {valueDisplay(props.row.value)}
            </span>

            <span class="text-stone-600 text-xs shrink-0">{typeMeta(props.row.value)}</span>

            <button
                onClick={copy}
                class={`shrink-0 flex items-center rounded-sm px-1 py-0.5 transition-colors
                    ${copied()
                        ? "text-emerald-400 bg-stone-700"
                        : "text-stone-700 hover:text-stone-300 hover:bg-stone-700 group-hover:text-stone-500"
                    }`}
                title="Copy value"
            >
                <Show when={copied()} fallback={<Icon.Clipboard size={17} stroke="currentColor" stroke-width={2} />}>
                    <Icon.Check size={17} stroke="currentColor" stroke-width={2} />
                </Show>
            </button>

            <button
                onClick={openSave}
                onKeyDown={(e) => { if (e.key === "Escape") { e.stopPropagation(); setSaving(false); } }}
                class={`shrink-0 flex items-center rounded-sm px-1 py-0.5 transition-colors
                    ${saving()
                        ? "text-amber-400 bg-stone-700"
                        : "text-stone-700 hover:text-amber-500 hover:bg-stone-700 group-hover:text-stone-500"
                    }`}
                title="Save as placeholder"
            >
                <Icon.ChevronsRight size={18} stroke="currentColor" stroke-width={2} />
            </button>

            <Show when={saving()}>
                <Show when={loading()} fallback={
                    <span class="inline-flex items-center gap-2 shrink-0">
                        <input
                            ref={inputEl!}
                            class="bg-stone-700 text-stone-100 text-sm font-mono
                                   rounded px-2.5 outline-none w-32 h-6"
                            value={name()}
                            onInput={(e) => setName(e.currentTarget.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter") confirm();
                                if (e.key === "Escape") { e.stopPropagation(); setSaving(false); }
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

// Renders nested object/array content when expanded. Separate component so
// onMount fires on each Show remount (i.e. each time user expands the node).
function NestedExpanded(props: { value: unknown; backend: Backend }) {
    const rows: FlatRow[] = [];
    if (Array.isArray(props.value)) {
        (props.value as unknown[]).forEach((v, i) => flatten(v, 1, String(i), rows));
    } else {
        Object.entries(props.value as Record<string, unknown>).forEach(([k, v]) =>
            flatten(v, 1, k, rows)
        );
    }

    let containerEl!: HTMLDivElement;
    onMount(() => { animate(containerEl, { opacity: [0, 1] }, { duration: 0.1 }); });

    return (
        <div ref={containerEl} class="pl-3 border-l border-stone-800 ml-2 mb-0.5">
            <For each={rows}>
                {(row) => <FlatJsonRow row={row} rowClass="" visible backend={props.backend} />}
            </For>
        </div>
    );
}

// A top-level field whose value is an object or array — collapsed by default.
function NestedToggleRow(props: { fieldKey: string; value: unknown; backend: Backend }) {
    const [open, setOpen] = createSignal(false);

    return (
        <div>
            <button
                onClick={() => setOpen(o => !o)}
                class="flex items-center gap-1.5 h-6 w-full text-left group"
            >
                <Icon.ChevronDown
                    size={13}
                    stroke="currentColor"
                    stroke-width={2}
                    style={{
                        transform: open() ? "rotate(0deg)" : "rotate(-90deg)",
                        transition: "transform 0.12s",
                        color: "currentColor",
                    }}
                    class="text-stone-600 group-hover:text-stone-400 shrink-0"
                />
                <span class="text-stone-400 font-mono text-xs shrink-0">"{props.fieldKey}":</span>
                <span class="text-stone-600 text-xs shrink-0">{typeMeta(props.value)}</span>
            </button>
            <Show when={open()}>
                <NestedExpanded value={props.value} backend={props.backend} />
            </Show>
        </div>
    );
}

// The expanded body of an accordion record. Separate component so onMount
// fires on each Show remount, triggering the fade animation per expansion.
// Rows start visible so badge toggles (reactive For updates) appear immediately.
function ExpandedRecord(props: {
    item: unknown;
    schemaKeys: string[];
    focusedFields: () => Set<string>;
    backend: Backend;
}) {
    let containerEl!: HTMLDivElement;
    onMount(() => { animate(containerEl, { opacity: [0, 1] }, { duration: 0.12 }); });

    const obj = (typeof props.item === "object" && props.item !== null && !Array.isArray(props.item))
        ? (props.item as Record<string, unknown>)
        : null;

    // No focus = show all; some focused = show only those
    const visibleKeys = () => {
        const focused = props.focusedFields();
        return focused.size > 0
            ? props.schemaKeys.filter(k => focused.has(k))
            : props.schemaKeys;
    };

    return (
        <div ref={containerEl} class="pl-3 pb-1 flex flex-col border-l border-stone-800 ml-3 mb-0.5">
            <Show when={obj !== null} fallback={
                <FlatJsonRow
                    row={{ depth: 0, label: null, value: props.item }}
                    rowClass="" visible
                    backend={props.backend}
                />
            }>
                <For each={visibleKeys()}>
                    {(key) => {
                        if (!(key in obj!)) return null;
                        const val = obj![key];
                        const isNested = val !== null && typeof val === "object";
                        return isNested
                            ? <NestedToggleRow fieldKey={key} value={val} backend={props.backend} />
                            : <FlatJsonRow row={{ depth: 0, label: key, value: val }} rowClass="" visible={true} backend={props.backend} />;
                    }}
                </For>
            </Show>
        </div>
    );
}

// One row in the accordion list — collapsed summary + expandable detail.
function AccordionRecord(props: {
    item: unknown;
    open: boolean;
    onToggle: () => void;
    schemaKeys: string[];
    focusedFields: () => Set<string>;
    outerCls: string;
    backend: Backend;
}) {
    const summary = summarizeRecord(props.item);

    return (
        <div>
            <div
                class={`${props.outerCls}-hdr flex items-center gap-2 h-7 px-1 cursor-pointer select-none rounded hover:bg-stone-800/50`}
                style={{ opacity: 0 }}
                onClick={props.onToggle}
            >
                <Icon.ChevronDown
                    size={14}
                    stroke="currentColor"
                    stroke-width={2}
                    style={{
                        transform: props.open ? "rotate(0deg)" : "rotate(-90deg)",
                        transition: "transform 0.12s",
                    }}
                    class="text-stone-600 shrink-0"
                />
                <span class="font-mono text-xs text-stone-400 truncate">{summary}</span>
            </div>
            <Show when={props.open}>
                <ExpandedRecord
                    item={props.item}
                    schemaKeys={props.schemaKeys}
                    focusedFields={props.focusedFields}
                    backend={props.backend}
                />
            </Show>
        </div>
    );
}

export function ResultView(props: { result: string | null; backend: Backend }) {
    // Signals must be declared before any conditional returns
    const [focusedFields, setFocusedFields] = createSignal(new Set<string>());
    const [openRecords, setOpenRecords] = createSignal(new Set<number>());
    const cls = `jr-${Math.random().toString(36).slice(2, 7)}`;

    function toggleFocus(key: string) {
        setFocusedFields(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key); else next.add(key);
            return next;
        });
    }

    function toggleRecord(i: number) {
        setOpenRecords(prev => {
            const next = new Set(prev);
            if (next.has(i)) next.delete(i); else next.add(i);
            return next;
        });
    }

    if (!props.result) return null;

    let parsed: unknown;
    try {
        parsed = JSON.parse(props.result);
    } catch {
        return <pre class="text-red-400 text-sm font-mono px-3 py-2">{props.result}</pre>;
    }

    const isArray = Array.isArray(parsed);
    const items = isArray ? (parsed as unknown[]) : [];
    const schemaKeys = isArray ? extractSchemaKeys(items) : [];
    const useAccordion = isArray && schemaKeys.length > 0;

    // Build flat rows only for non-accordion path
    const rows: FlatRow[] = [];
    if (!useAccordion) {
        if (isArray) {
            items.forEach((v, i) => flatten(v, 0, String(i), rows));
        } else {
            flatten(parsed, 0, null, rows);
        }
    }

    onMount(() => {
        if (!useAccordion) {
            const scroller = document.getElementById("scroll-root");
            const rowEls = Array.from(document.querySelectorAll(`.${cls}`)) as HTMLElement[];

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const seq: any[] = [];
            rowEls.forEach((el, i) => {
                const t = i * 0.05;
                seq.push([el, { opacity: 1 }, { duration: 0.05, at: t }]);
                seq.push([() => {
                    if (!scroller) return;
                    const rect = el.getBoundingClientRect();
                    const scrollerRect = scroller.getBoundingClientRect();
                    const target = scrollerRect.top + scrollerRect.height * 0.3;
                    const delta = (rect.top + rect.height / 2) - target;
                    if (delta > 0) scroller.scrollTop += delta;
                }, { at: t + 0.05 }]);
            });
            const finalT = rowEls.length > 0 ? (rowEls.length - 1) * 0.05 + 0.06 : 0.06;
            seq.push([() => focusComposing(), { at: finalT }]);
            animate(seq);
        } else {
            // Accordion: stagger header rows in with same scroll behaviour as flat view
            const scroller = document.getElementById("scroll-root");
            const hdrEls = Array.from(document.querySelectorAll(`.${cls}-hdr`)) as HTMLElement[];
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const seq: any[] = [];
            hdrEls.forEach((el, i) => {
                const t = i * 0.04;
                seq.push([el, { opacity: 1 }, { duration: 0.04, at: t }]);
                seq.push([() => {
                    if (!scroller) return;
                    const rect = el.getBoundingClientRect();
                    const scrollerRect = scroller.getBoundingClientRect();
                    const target = scrollerRect.top + scrollerRect.height * 0.3;
                    const delta = (rect.top + rect.height / 2) - target;
                    if (delta > 0) scroller.scrollTop += delta;
                }, { at: t + 0.04 }]);
            });
            const finalT = hdrEls.length > 0 ? (hdrEls.length - 1) * 0.04 + 0.05 : 0.05;
            seq.push([() => focusComposing(), { at: finalT }]);
            if (seq.length > 1) animate(seq); else focusComposing();
        }
    });

    if (useAccordion) {
        const allOpen = () => openRecords().size === items.length;
        const expandAll = () => setOpenRecords(new Set(items.map((_, i) => i)));
        const collapseAll = () => setOpenRecords(new Set());

        return (
            <div class="px-3 py-2 flex flex-col gap-px">
                <div class="flex items-start justify-between gap-3 pb-2 mb-1 border-b border-stone-800">
                    <div class="flex flex-wrap gap-1">
                        <For each={schemaKeys}>
                            {(key) => (
                                <button
                                    onClick={() => toggleFocus(key)}
                                    class={`px-2 py-0.5 rounded text-xs font-mono border transition-colors ${
                                        focusedFields().size === 0
                                            ? "text-stone-500 border-stone-700 hover:text-stone-300 hover:border-stone-500"
                                            : focusedFields().has(key)
                                                ? "text-amber-400 bg-amber-400/10 border-amber-400/30"
                                                : "text-stone-700 border-stone-800"
                                    }`}
                                >
                                    {key}
                                </button>
                            )}
                        </For>
                    </div>
                    <div class="flex items-center gap-2 shrink-0 pt-px">
                        <button
                            onClick={allOpen() ? collapseAll : expandAll}
                            class="text-xs text-stone-600 hover:text-stone-400 transition-colors shrink-0"
                        >
                            {allOpen() ? "collapse all" : "expand all"}
                        </button>
                    </div>
                </div>
                <For each={items}>
                    {(item, i) => (
                        <AccordionRecord
                            item={item}
                            open={openRecords().has(i())}
                            onToggle={() => toggleRecord(i())}
                            schemaKeys={schemaKeys}
                            focusedFields={focusedFields}
                            outerCls={cls}
                            backend={props.backend}
                        />
                    )}
                </For>
                <Show when={items.length === 0}>
                    <span class="text-stone-600 text-sm italic px-1 py-1">[]</span>
                </Show>
            </div>
        );
    }

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
