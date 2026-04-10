import {createSignal, Show} from "solid-js";
import {Controls} from "./controls.tsx";
import {Block} from "../bindings/Block.ts";
import {Backend} from "../backend.tsx";

export function EditKeybindDisplay() {
    return <>
    </>
}
export function BlockView(data: { block: Block, backend: Backend }) {
    const composing = createSignal(true);
    return <>
        <div class={"w-full rounded-sm bg-stone-800 min-h-24 p-2"}>
            <div class={"text-stone-500 text-sm"}>{"ctx:user - "}{" specs / metrics / ... "}</div>
            <div class={"w-full h-full"}>
                <textarea class={"outline-none w-full min-h-24 text-stone-300  resize-none rounded-sm"}></textarea>
            </div>
        </div>
        <Show when={composing[0]()} fallback={<EditKeybindDisplay />}>
            <Controls placeholders={data.backend.placeholders()} ids={data.backend.ids()}
                      schema={data.backend.schema()}/>
        </Show>
    </>
}