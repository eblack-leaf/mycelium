import {createSignal, Show} from "solid-js";
import {Controls} from "./controls.tsx";

export type Block = {
    query: string,
};
export function BlockView(_props: { data: Block, getter: () => string, setter: (str: string) => void }) {
    const composing = createSignal(true);
    return <>
        <div class={"w-full rounded-sm bg-stone-800 min-h-24 "}>
            <div class={""}></div>
            <Show when={composing[0]()} fallback={<></>}>
                <Controls/>
            </Show>
        </div>
    </>
}