const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");

export type StockInfo = {
  code: string;
  name: string;
  is_trained: boolean;
};

export type TrainedStocksResponse = {
  stocks: StockInfo[];
};

export type PredictSuccessResponse = {
  success: true;
  code: string;
  name: string;
  is_trained: true;
  buy_count: number;
  total_models: number;
  recommendation: string;
  summary: string;
  models: {
    xgb: "上漲" | "下跌";
    rnn: "上漲" | "下跌";
    lstm: "上漲" | "下跌";
    transformer: "上漲" | "下跌";
  };
};

export type PredictUntrainedResponse = {
  success: true;
  code: string;
  name: string;
  is_trained: false;
  message: string;
};

export type PredictErrorResponse = {
  success: false;
  error_type: string;
  message: string;
  code?: string;
};

export type PredictResponse =
  | PredictSuccessResponse
  | PredictUntrainedResponse
  | PredictErrorResponse;

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const data = (await response.json()) as T;
  return data;
}

export async function getTrainedStocks(): Promise<TrainedStocksResponse> {
  const response = await fetch(`${API_BASE_URL}/api/trained-stocks`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error("取得已訓練股票清單失敗");
  }

  return parseJsonResponse<TrainedStocksResponse>(response);
}

export async function predictStock(code: string): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code })
  });

  return parseJsonResponse<PredictResponse>(response);
}
