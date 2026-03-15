import { createSignal, onCleanup, onMount } from "solid-js";

const PHRASES = [
  "show me all users created this week",
  "which orders have no shipment yet?",
  "count records in sessions grouped by day",
  "find products where stock < 10",
  "give me the last 5 errors from logs",
];

const TYPE_MS = 110;
const DELETE_MS = 38;
const PAUSE_MS = 3200;

function usePlaceholderTyper() {
  const [text, setText] = createSignal("");
  let phraseIdx = 0;
  let charIdx = 0;
  let deleting = false;
  let timer: ReturnType<typeof setTimeout>;

  function tick() {
    const phrase = PHRASES[phraseIdx];
    if (!deleting) {
      charIdx++;
      setText(phrase.slice(0, charIdx));
      if (charIdx === phrase.length) {
        deleting = true;
        timer = setTimeout(tick, PAUSE_MS);
        return;
      }
      timer = setTimeout(tick, TYPE_MS);
    } else {
      charIdx--;
      setText(phrase.slice(0, charIdx));
      if (charIdx === 0) {
        deleting = false;
        phraseIdx = (phraseIdx + 1) % PHRASES.length;
        timer = setTimeout(tick, TYPE_MS);
        return;
      }
      timer = setTimeout(tick, DELETE_MS);
    }
  }

  onMount(() => { timer = setTimeout(tick, 600); });
  onCleanup(() => clearTimeout(timer));

  return text;
}

export function QueryBar() {
  const [input, setInput] = createSignal("");
  const placeholder = usePlaceholderTyper();

  function handleInput(e: Event) {
    const el = e.currentTarget as HTMLTextAreaElement;
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
    setInput(el.value);
  }

  return (
    <textarea
      rows={2}
      placeholder={placeholder()}
      onInput={handleInput}
      value={input()}
      class="w-full resize-none overflow-hidden rounded-2xl bg-stone-700 text-neutral-100 placeholder-neutral-400 px-4 py-3 text-sm leading-relaxed outline-none"
      style={{ "min-height": "92px", "max-height": "240px" }}
    />
  );
}
