import type { StockInfo } from "@/lib/api";

type StockChipsProps = {
  stocks: StockInfo[];
  onSelect: (stock: StockInfo) => void;
};

export default function StockChips({ stocks, onSelect }: StockChipsProps) {
  return (
    <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-700">目前可預測的個股</h2>
        <span className="text-xs text-slate-400">資料來自後端 /api/trained-stocks</span>
      </div>

      <div className="flex flex-wrap gap-2">
        {stocks.map((stock) => (
          <button
            key={stock.code}
            type="button"
            onClick={() => onSelect(stock)}
            className={[
              "rounded-full border px-4 py-2 text-sm transition",
              stock.is_trained
                ? "border-slate-300 bg-slate-900 text-white hover:bg-slate-700"
                : "border-slate-200 bg-slate-100 text-slate-400 hover:bg-slate-200"
            ].join(" ")}
            aria-label={`選擇 ${stock.code} ${stock.name}`}
          >
            {stock.code} {stock.name}
            {!stock.is_trained && <span className="ml-2 text-xs">尚未訓練</span>}
          </button>
        ))}
      </div>
    </section>
  );
}
