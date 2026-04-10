import { createEffect, createMemo, For } from "solid-js";

interface TextSegment {
    text: string;
    kind: "plain" | "placeholder" | "keyword";
}

function tokenize(text: string, prefix: string): TextSegment[] {
    if (!text) return [{ text: "", kind: "plain" }];

    const escapedPrefix = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Match placeholder tokens OR keyword tokens, left to right
    const placeholderRe = new RegExp(`${escapedPrefix}[\\w-]+`, "g");
    const keywordRe = new RegExp(`\\b(${SURREAL_KEYWORDS.join("|")})\\b`, "gi");

    // Build a sorted list of all matches with their kind
    type Match = { start: number; end: number; kind: "placeholder" | "keyword" };
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
    onArrowNav: (dir: "up" | "down" | "left" | "right") => void;
    onHistory: (dir: "up" | "down") => void;
    ref?: (r: HighlightedTextareaRef) => void;
}

export function HighlightedTextarea(props: Props) {
    let textareaEl!: HTMLTextAreaElement;

    const segments = createMemo(() => tokenize(props.value, props.prefix));

    // Sync textarea scroll with the mirror div
    let mirrorEl!: HTMLDivElement;
    createEffect(() => {
        // Re-run whenever value changes to keep mirror scroll in sync
        props.value;
    });

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
        } else if (e.key === "ArrowDown") {
            e.preventDefault();
            props.onArrowNav("down");
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            props.onArrowNav("up");
        } else if (e.key === "ArrowLeft" && e.altKey) {
            e.preventDefault();
            props.onArrowNav("left");
        } else if (e.key === "ArrowRight" && e.altKey) {
            e.preventDefault();
            props.onArrowNav("right");
        }
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
            {/* Mirror layer — colored spans */}
            <div
                ref={mirrorEl!}
                aria-hidden="true"
                class="absolute inset-0 whitespace-pre-wrap break-words pointer-events-none
                       text-stone-300 p-0 overflow-hidden"
            >
                <For each={segments()}>
                    {(seg) => (
                        <span
                            class={
                                seg.kind === "placeholder"
                                    ? "text-amber-400"
                                    : seg.kind === "keyword"
                                    ? "text-orange-400"
                                    : "text-stone-300"
                            }
                        >
                            {seg.text}
                        </span>
                    )}
                </For>
                {/* Invisible trailing character to prevent mirror height collapse */}
                <span class="invisible"> </span>
            </div>
            {/* Actual textarea — transparent text, visible caret */}
            <textarea
                ref={textareaEl!}
                value={props.value}
                onInput={(e) => props.onChange(e.currentTarget.value)}
                onKeyDown={onKeyDown}
                spellcheck={false}
                autocomplete="off"
                class="relative w-full bg-transparent caret-white text-transparent
                       resize-none outline-none font-mono text-sm leading-relaxed
                       whitespace-pre-wrap break-words min-h-16"
                style={{ "min-height": "4rem" }}
                rows={1}
            />
        </div>
    );
}

const SURREAL_KEYWORDS = [
    "SELECT", "CREATE", "UPDATE", "DELETE", "RELATE", "RETURN", "INSERT",
    "UPSERT", "DEFINE", "REMOVE", "INFO", "USE", "LET", "IF", "ELSE",
    "FOR", "BREAK", "CONTINUE", "BEGIN", "COMMIT", "CANCEL", "THROW",
    "SLEEP", "SHOW", "LIVE", "KILL",
    "FROM", "WHERE", "SET", "MERGE", "CONTENT", "REPLACE", "UNSET",
    "LIMIT", "ORDER", "GROUP", "SPLIT", "FETCH", "START", "BY", "ONLY",
    "WITH", "TIMEOUT", "PARALLEL", "EXPLAIN",
    "ASC", "DESC", "AND", "OR", "NOT", "IS", "IN", "NONE", "NULL",
    "TRUE", "FALSE", "TYPE", "ASSERT", "VALUE", "DEFAULT", "READONLY",
    "PERMISSIONS", "FLEXIBLE", "SCHEMAFULL", "SCHEMALESS",
    "ON", "FIELD", "INDEX", "TABLE", "SCOPE", "PARAM", "FUNCTION",
    "UNIQUE", "SEARCH", "ANALYZER", "NAMESPACE", "DATABASE",
];
