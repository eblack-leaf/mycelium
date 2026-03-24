export function Block(props: { data: BlockData }) {
    return <>
        <div class={"bg-stone-700 rounded-sm w-full h-12"}>
            <div>{props.data.query}</div>
        </div>
    </>
}

export type BlockData = {
    query: string,
}