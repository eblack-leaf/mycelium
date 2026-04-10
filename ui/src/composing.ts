// Shared ref to the current composing block's textarea + scroll container.
// Set by the composing BlockView on mount/update, cleared when it unmounts.

let textareaEl: HTMLTextAreaElement | null = null;
let containerEl: HTMLElement | null = null;

export function setComposingEls(
    textarea: HTMLTextAreaElement,
    container: HTMLElement,
) {
    textareaEl = textarea;
    containerEl = container;
}

export function clearComposingEls() {
    textareaEl = null;
    containerEl = null;
}

/** Focus without moving the viewport — called after result animation finishes. */
export function focusComposing() {
    textareaEl?.focus({ preventScroll: true });
}

/** Scroll composing block to top of viewport then focus — called by global Escape. */
export function jumpToComposing() {
    containerEl?.scrollIntoView({ behavior: "smooth", block: "start" });
    textareaEl?.focus({ preventScroll: true });
}
