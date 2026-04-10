import { createMemo, For } from "solid-js";

interface TextSegment {
    text: string;
    kind: "plain" | "placeholder" | "keyword" | "operator";
}

// SurrealDB-specific operators and graph traversal symbols
const OPERATOR_RE = /<->|->|<-|<\+=|>=|<=|!=|\*=|\?=|\+=|-=|~=|!~|::|\.\.\.|\.\.|=>/g;

function tokenize(text: string, prefix: string): TextSegment[] {
    if (!text) return [{ text: "", kind: "plain" }];

    const escapedPrefix = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const placeholderRe = new RegExp(`${escapedPrefix}[\\w-]+`, "g");
    const keywordRe = new RegExp(`\\b(${SURREAL_KEYWORDS.join("|")})\\b`, "gi");
    const operatorRe = new RegExp(OPERATOR_RE.source, "g");

    type Match = { start: number; end: number; kind: "placeholder" | "keyword" | "operator" };
    const matches: Match[] = [];

    let m: RegExpExecArray | null;
    placeholderRe.lastIndex = 0;
    while ((m = placeholderRe.exec(text)) !== null) {
        matches.push({ start: m.index, end: m.index + m[0].length, kind: "placeholder" });
    }
    keywordRe.lastIndex = 0;
    while ((m = keywordRe.exec(text)) !== null) {
        matches.push({ start: m.index, end: m.index + m[0].length, kind: "keyword" });
    }
    operatorRe.lastIndex = 0;
    while ((m = operatorRe.exec(text)) !== null) {
        matches.push({ start: m.index, end: m.index + m[0].length, kind: "operator" });
    }

    // Sort by start position; remove overlaps (placeholder wins)
    matches.sort((a, b) => a.start - b.start);
    const nonOverlapping: Match[] = [];
    let cursor = 0;
    for (const match of matches) {
        if (match.start < cursor) continue; // overlapping — skip
        nonOverlapping.push(match);
        cursor = match.end;
    }

    // Build segments
    const segments: TextSegment[] = [];
    let pos = 0;
    for (const match of nonOverlapping) {
        if (match.start > pos) {
            segments.push({ text: text.slice(pos, match.start), kind: "plain" });
        }
        segments.push({ text: text.slice(match.start, match.end), kind: match.kind });
        pos = match.end;
    }
    if (pos < text.length) {
        segments.push({ text: text.slice(pos), kind: "plain" });
    }
    return segments.length ? segments : [{ text, kind: "plain" }];
}

export interface HighlightedTextareaRef {
    insertAt(from: number, to: number, text: string): void;
    getCursorPos(): number;
}

interface Props {
    value: string;
    onChange: (v: string) => void;
    prefix: string;
    onSubmit: () => void;
    onTab: () => void;
    onSelectCompletion: (index: number) => void;
    onHistory: (dir: "up" | "down") => void;
    pasteTransform?: (text: string) => Promise<string>;
    ref?: (r: HighlightedTextareaRef) => void;
}

export function HighlightedTextarea(props: Props) {
    let textareaEl!: HTMLTextAreaElement;

    const segments = createMemo(() => tokenize(props.value, props.prefix));

    let mirrorEl!: HTMLDivElement;

    function onKeyDown(e: KeyboardEvent) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            props.onSubmit();
        } else if (e.key === "Tab") {
            e.preventDefault();
            props.onTab();
        } else if (e.key === "ArrowDown" && e.altKey) {
            e.preventDefault();
            props.onHistory("down");
        } else if (e.key === "ArrowUp" && e.altKey) {
            e.preventDefault();
            props.onHistory("up");
        } else if (e.ctrlKey && e.key >= "1" && e.key <= "4") {
            e.preventDefault();
            props.onSelectCompletion(parseInt(e.key) - 1);
        }
    }

    async function pasteTransformHandler(e: ClipboardEvent) {
        if (!props.pasteTransform) return;
        e.preventDefault();
        const text = e.clipboardData?.getData("text") ?? "";
        if (!text) return;
        const replacement = await props.pasteTransform(text);
        const start = textareaEl.selectionStart ?? 0;
        const end = textareaEl.selectionEnd ?? start;
        const next = props.value.slice(0, start) + replacement + props.value.slice(end);
        props.onChange(next);
        requestAnimationFrame(() => {
            const pos = start + replacement.length;
            textareaEl.setSelectionRange(pos, pos);
        });
    }

    if (props.ref) {
        props.ref({
            insertAt(from: number, to: number, text: string) {
                const current = props.value;
                const next = current.slice(0, from) + text + current.slice(to);
                props.onChange(next);
                // Restore cursor after the inserted text
                requestAnimationFrame(() => {
                    const pos = from + text.length;
                    textareaEl.setSelectionRange(pos, pos);
                    textareaEl.focus();
                });
            },
            getCursorPos() {
                return textareaEl.selectionStart ?? 0;
            },
        });
    }

    return (
        <div class="relative font-mono text-sm leading-relaxed min-h-16">
            {/* Mirror layer — in normal flow, drives container height */}
            <div
                ref={mirrorEl!}
                aria-hidden="true"
                class="w-full whitespace-pre-wrap break-words pointer-events-none
                       text-stone-300 p-0"
            >
                <For each={segments()}>
                    {(seg) => (
                        <span
                            class={
                                seg.kind === "placeholder"
                                    ? "text-amber-400"
                                    : seg.kind === "keyword"
                                    ? "text-orange-400"
                                    : seg.kind === "operator"
                                    ? "text-sky-400"
                                    : "text-stone-300"
                            }
                        >
                            {seg.text}
                        </span>
                    )}
                </For>
                {/* Trailing space keeps height from collapsing on empty value */}
                <span class="invisible"> </span>
            </div>
            {/* Actual textarea — absolute over mirror, transparent text, visible caret */}
            <textarea
                ref={textareaEl!}
                value={props.value}
                onInput={(e) => props.onChange(e.currentTarget.value)}
                onKeyDown={onKeyDown}
                onPaste={pasteTransformHandler}
                spellcheck={false}
                autocomplete="off"
                class="absolute inset-0 w-full h-full bg-transparent caret-white text-transparent
                       resize-none outline-none p-0 font-mono text-sm leading-relaxed
                       whitespace-pre-wrap break-words overflow-hidden"
            />
        </div>
    );
}

const SURREAL_KEYWORDS = [
    // Statements
    "SELECT", "CREATE", "UPDATE", "DELETE", "RELATE", "RETURN", "INSERT",
    "UPSERT", "DEFINE", "REMOVE", "INFO", "USE", "LET", "IF", "ELSE",
    "THEN", "END", "FOR", "BREAK", "CONTINUE", "BEGIN", "COMMIT", "CANCEL",
    "THROW", "SLEEP", "SHOW", "LIVE", "KILL", "REBUILD", "OPTION",
    // Clauses
    "FROM", "WHERE", "SET", "MERGE", "CONTENT", "REPLACE", "UNSET",
    "LIMIT", "ORDER", "GROUP", "SPLIT", "FETCH", "START", "BY", "ONLY",
    "WITH", "TIMEOUT", "PARALLEL", "EXPLAIN", "TEMPFILES", "OMIT",
    "BEFORE", "AFTER", "DIFF", "WHEN", "OVERWRITE", "NOINDEX",
    // Operators / logic
    "ASC", "DESC", "AND", "OR", "NOT", "IS", "IN", "NONE", "NULL",
    "CONTAINS", "CONTAINSALL", "CONTAINSANY", "CONTAINSNONE",
    "INSIDE", "NOTINSIDE", "ALLINSIDE", "ANYINSIDE", "NONEINSIDE",
    "OUTSIDE", "INTERSECTS",
    // Values
    "TRUE", "FALSE", "FUTURE",
    // Schema
    "TYPE", "ASSERT", "VALUE", "DEFAULT", "READONLY", "FLEXIBLE",
    "PERMISSIONS", "SCHEMAFULL", "SCHEMALESS", "ENFORCED",
    "ON", "FIELD", "INDEX", "TABLE", "SCOPE", "PARAM", "FUNCTION",
    "UNIQUE", "SEARCH", "ANALYZER", "NAMESPACE", "DATABASE",
    "EVENT", "RELATION", "REFERENCES",
    // Types
    "ANY", "ARRAY", "BOOL", "BYTES", "DATETIME", "DECIMAL", "DURATION",
    "FLOAT", "GEOMETRY", "INT", "NUMBER", "OBJECT", "RECORD", "STRING",
    "UUID",
    // Auth
    "SIGNIN", "SIGNUP", "AUTHENTICATE", "TOKEN", "SESSION",
];
