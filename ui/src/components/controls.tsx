import {For} from "solid-js";
import {Suggestion} from "../bindings/Suggestion.ts";

function SuggestionView(data: { suggestion: Suggestion }) {
    return <div class={"flex justify-end p-2"}>
        <div>{data.suggestion.text}</div>
        <div>{data.suggestion.metadata}</div>
    </div>
}

function ControlSuggestion(data: { items: Suggestion[] | undefined }) {
    return <div class={"flex flex-col gap-4"}>
        <For each={data.items}>
            {(ph) => {
                return <SuggestionView suggestion={ph}></SuggestionView>;
            }}
        </For>
    </div>
}

export function Controls(data: {
    placeholders: Suggestion[] | undefined,
    ids: Suggestion[] | undefined,
    schema: Suggestion[] | undefined
}) {
    return <>
        <div class={"grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 grid-rows-1 bg-none text-stone-400 gap-4 p-2"}>
            <ControlSuggestion items={data.placeholders}/>
            <ControlSuggestion items={data.ids}/>
            <ControlSuggestion items={data.schema}/>
        </div>
    </>
}