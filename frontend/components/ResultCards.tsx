import { AlertTriangle, ArrowLeft, BarChart3, CheckCircle2, HelpCircle, Loader2, RefreshCcw, SearchX } from "lucide-react";
import type { PredictSuccessResponse, PredictUntrainedResponse } from "@/lib/api";

export function IdleCard() {
  return (
    <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-500">
      <HelpCircle className="mx-auto mb-3 h-8 w-8 text-slate-400" />
      <p className="font-medium">尚未查詢</p>
      <p className="mt-1 text-sm">輸入 4 位數股票代號後，開始模型預測。</p>
    </div>
  );
}

export function LoadingCard() {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
      <Loader2 className="mx-auto mb-3 h-9 w-9 animate-spin text-slate-900" />
      <p className="text-lg font-semibold text-slate-900">模型推論中</p>
      <p className="mt-1 text-sm text-slate-500">正在呼叫後端 API 並載入 checkpoints 權重。</p>
    </div>
  );
}

export function ResultCard({ result, onReset }: { result: PredictSuccessResponse; onReset: () => void }) {
  const isBuy = result.buy_count >= 2;
  const modelLabelMap: Record<string, string> = {
    xgb: "XGBoost",
    rnn: "RNN",
    lstm: "LSTM",
    transformer: "Transformer"
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
        <div>
          <p className="text-sm text-slate-500">預測標的</p>
          <h2 className="text-2xl font-bold text-slate-900">
            {result.code} {result.name}
          </h2>
        </div>
        <span
          className={[
            "inline-flex w-fit rounded-full px-4 py-2 text-sm font-semibold",
            isBuy ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
          ].join(" ")}
        >
          {result.recommendation}
        </span>
      </div>

      <div className={[
        "mt-5 rounded-2xl p-5",
        isBuy ? "bg-red-50" : "bg-green-50"
      ].join(" ")}
      >
        <div className="flex items-start gap-3">
          <BarChart3 className={isBuy ? "mt-1 h-6 w-6 text-red-600" : "mt-1 h-6 w-6 text-green-600"} />
          <div>
            <p className="text-lg font-semibold text-slate-900">{result.buy_count} / {result.total_models} 個模型看漲</p>
            <p className="mt-1 text-sm text-slate-700">{result.summary}</p>
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        {Object.entries(result.models).map(([key, value]) => {
          const up = value === "上漲";
          return (
            <div key={key} className="rounded-xl border border-slate-200 p-4">
              <div className="flex items-center justify-between gap-3">
                <span className="font-medium text-slate-700">{modelLabelMap[key] || key}</span>
                <span className={up ? "font-semibold text-red-600" : "font-semibold text-green-600"}>{value}</span>
              </div>
            </div>
          );
        })}
      </div>

      <button
        type="button"
        onClick={onReset}
        className="mt-6 inline-flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700"
      >
        <ArrowLeft className="h-4 w-4" />
        查詢其他個股
      </button>
    </div>
  );
}

export function UntrainedCard({ result, onReset }: { result: PredictUntrainedResponse; onReset: () => void }) {
  return (
    <div className="rounded-2xl border border-blue-100 bg-white p-6 shadow-sm">
      <div className="flex items-start gap-4">
        <div className="rounded-full bg-blue-100 p-3">
          <CheckCircle2 className="h-6 w-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-900">尚未訓練 {result.code} {result.name} 的模型</h2>
          <p className="mt-2 text-sm text-slate-600">需先以歷史股價完成訓練後，才能進行隔日沖預測。</p>
          <p className="mt-2 text-sm text-slate-500">{result.message}</p>
          <button
            type="button"
            onClick={onReset}
            className="mt-5 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-500"
          >
            返回查詢
          </button>
        </div>
      </div>
    </div>
  );
}

export function UnsupportedCard({ message, onReset }: { message: string; onReset: () => void }) {
  return (
    <div className="rounded-2xl border border-amber-100 bg-white p-6 shadow-sm">
      <div className="flex items-start gap-4">
        <div className="rounded-full bg-amber-100 p-3">
          <SearchX className="h-6 w-6 text-amber-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-900">{message || "目前尚未支援此股票代號"}</h2>
          <p className="mt-2 text-sm text-slate-600">目前僅支援 2303、2344、3481、6770、8028。</p>
          <button
            type="button"
            onClick={onReset}
            className="mt-5 rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700"
          >
            返回查詢
          </button>
        </div>
      </div>
    </div>
  );
}

export function ErrorCard({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="rounded-2xl border border-red-100 bg-white p-6 shadow-sm">
      <div className="flex items-start gap-4">
        <div className="rounded-full bg-red-100 p-3">
          <AlertTriangle className="h-6 w-6 text-red-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-900">發生錯誤</h2>
          <p className="mt-2 text-sm text-slate-600">{message || "後端連線失敗，請確認 API server 是否已啟動。"}</p>
          <button
            type="button"
            onClick={onRetry}
            className="mt-5 inline-flex items-center gap-2 rounded-xl bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-500"
          >
            <RefreshCcw className="h-4 w-4" />
            重試
          </button>
        </div>
      </div>
    </div>
  );
}
