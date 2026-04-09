import {createSignal, Show} from "solid-js";
import {Controls} from "./controls.tsx";
import {invoke} from "@tauri-apps/api/core";
import {createResource} from "solid-js";
import {Suggestion} from "../bindings/Suggestion.ts";
import {Block} from "../bindings/Block.ts";


export function BlockView(_data: { block: Block }) {
    const composing = createSignal(true);
    const [placeholders] = createResource(async () => await invoke<Suggestion[]>("placeholders"));
    const [ids] = createResource(async () => await invoke<Suggestion[]>("ids"));
    const [schema] = createResource(async () => await invoke<Suggestion[]>("schema"));
    return <>
        <div class={"text-stone-500 text-sm"}>{"ctx:user - "}{" specs / metrics / ... "}</div>
        <div class={"w-full rounded-sm bg-stone-800 min-h-24"}>
            <div class={"w-full h-full p-2"}>
                <textarea class={"outline-none w-full min-h-24 text-stone-300  resize-none rounded-sm"}></textarea>
            </div>
        </div>
        <Show when={composing[0]()} fallback={<></>}>
            <Controls placeholders={placeholders()} ids={ids()}
                      schema={schema()}/>
        </Show>
    </>
}