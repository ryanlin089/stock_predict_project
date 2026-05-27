"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, LineChart } from "lucide-react";

import { getTrainedStocks, predictStock, type PredictResponse, type PredictSuccessResponse, type PredictUntrainedResponse, type StockInfo } from "@/lib/api";
import StockChips from "@/components/StockChips";
import { ErrorCard, IdleCard, LoadingCard, ResultCard, UntrainedCard, UnsupportedCard } from "@/components/ResultCards";

type ViewState = "idle" | "loading" | "success" | "untrained" | "unsupported" | "error";

export default function StockPredictor() {
  const [code, setCode] = useState("");
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [viewState, setViewState] = useState<ViewState>("idle");
  const [successResult, setSuccessResult] = useState<PredictSuccessResponse | null>(null);
  const [untrainedResult, setUntrainedResult] = useState<PredictUntrainedResponse | null>(null);
  const [unsupportedMessage, setUnsupportedMessage] = useState("目前尚未支援此股票代號");
  const [errorMessage, setErrorMessage] = useState("");
  const [stockListError, setStockListError] = useState("");

  useEffect(() => {
    async function fetchStocks() {
      try {
        const response = await getTrainedStocks();
        setStocks(response.stocks);
      } catch (error) {
        setStockListError("無法取得已訓練股票清單，請確認後端 API server 是否已啟動。");
      }
    }

    fetchStocks();
  }, []);

  const hint = useMemo(() => {
    if (code.length === 0) return "請輸入 4 位數台股代號";
    if (code.length < 4) return `還需輸入 ${4 - code.length} 碼`;
    return "格式正確";
  }, [code]);

  const isValid = code.length === 4;

  function handleInputChange(value: string) {
    const numbersOnly = value.replace(/\D/g, "").slice(0, 4);
    setCode(numbersOnly);
  }

  function resetResult() {
    setViewState("idle");
    setSuccessResult(null);
    setUntrainedResult(null);
    setUnsupportedMessage("目前尚未支援此股票代號");
    setErrorMessage("");
  }

  function resolvePredictResponse(response: PredictResponse) {
    if (response.success && response.is_trained) {
      setSuccessResult(response);
      setViewState("success");
      return;
    }

    if (response.success && !response.is_trained) {
      setUntrainedResult(response);
      setViewState("untrained");
      return;
    }

    if (!response.success && response.error_type === "UNSUPPORTED_STOCK") {
      setUnsupportedMessage(response.message);
      setViewState("unsupported");
      return;
    }

    setErrorMessage(response.message || "後端連線失敗，請確認 API server 是否已啟動。");
    setViewState("error");
  }

  async function handlePredict(targetCode = code) {
    if (targetCode.length !== 4) return;

    setCode(targetCode);
    setViewState("loading");
    setSuccessResult(null);
    setUntrainedResult(null);
    setUnsupportedMessage("目前尚未支援此股票代號");
    setErrorMessage("");

    try {
      const response = await predictStock(targetCode);
      resolvePredictResponse(response);
    } catch (error) {
      setErrorMessage("後端連線失敗，請確認 API server 是否已啟動。");
      setViewState("error");
    }
  }

  function handleChipSelect(stock: StockInfo) {
    setCode(stock.code);

    if (stock.is_trained) {
      handlePredict(stock.code);
      return;
    }

    setUntrainedResult({
      success: true,
      code: stock.code,
      name: stock.name,
      is_trained: false,
      message: `尚未訓練 ${stock.code} ${stock.name} 的模型，請先完成模型訓練後再進行預測`
    });
    setViewState("untrained");
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" && isValid && viewState !== "loading") {
      handlePredict();
    }
  }

  return (
    <main className="min-h-screen px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-3xl">
        <header className="text-center">
          <div className="inline-flex items-center gap-3">
            <div className="rounded-2xl bg-slate-900 p-3 text-white shadow-sm">
              <LineChart className="h-8 w-8" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">隔日沖預測</h1>
          </div>
          <p className="mt-4 text-base text-slate-500 sm:text-lg">輸入台股代號，由模型預測該股明日漲跌傾向</p>
        </header>

        <section className="mt-10 rounded-3xl border border-slate-200 bg-slate-100 p-6 shadow-sm">
          <label htmlFor="stock-code" className="mb-3 block text-center text-sm font-medium text-slate-600">
            台股代號
          </label>
          <input
            id="stock-code"
            value={code}
            onChange={(event) => handleInputChange(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="2303"
            inputMode="numeric"
            className="w-full rounded-2xl border border-slate-300 bg-white px-5 py-5 text-center font-mono text-[34px] font-semibold tracking-[0.35em] text-slate-900 outline-none transition placeholder:tracking-normal placeholder:text-slate-300 focus:border-slate-900 focus:ring-4 focus:ring-slate-200"
          />
          <p className={isValid ? "mt-3 text-center text-sm font-medium text-green-600" : "mt-3 text-center text-sm text-slate-500"}>
            {hint}
          </p>
          <button
            type="button"
            onClick={() => handlePredict()}
            disabled={!isValid || viewState === "loading"}
            className="mt-5 w-full rounded-2xl bg-slate-900 px-5 py-4 text-base font-semibold text-white transition hover:bg-slate-700 disabled:bg-slate-300 disabled:text-slate-500"
          >
            開始預測
          </button>
        </section>

        {stockListError ? (
          <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-700">
            {stockListError}
          </div>
        ) : (
          <StockChips stocks={stocks} onSelect={handleChipSelect} />
        )}

        <section className="mt-8">
          {viewState === "idle" && <IdleCard />}
          {viewState === "loading" && <LoadingCard />}
          {viewState === "success" && successResult && <ResultCard result={successResult} onReset={resetResult} />}
          {viewState === "untrained" && untrainedResult && <UntrainedCard result={untrainedResult} onReset={resetResult} />}
          {viewState === "unsupported" && <UnsupportedCard message={unsupportedMessage} onReset={resetResult} />}
          {viewState === "error" && <ErrorCard message={errorMessage} onRetry={() => handlePredict()} />}
        </section>

        <footer className="mt-10 rounded-2xl bg-slate-200/70 p-4 text-sm text-slate-600">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-slate-500" />
            <p>本預測僅供學術研究參考，不構成投資建議，投資決策請自行評估。</p>
          </div>
        </footer>
      </div>
    </main>
  );
}
