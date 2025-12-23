#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelCategory {
    PDE, // 偏微分方程式 (空間がある: Wave, Heat)
    ODE, // 常微分方程式 (空間がない: Lorenz, Neural ODE)
}

/// Dyneシステム上で動作する物理エンジンの共通インターフェース
pub trait DyneEngine {
    /// 1ステップ時間を進める
    fn step(&mut self, input: &[f32]) -> Vec<f32>;

    /// このモデルが PDE か ODE かを返す
    fn category(&self) -> ModelCategory;

    /// 境界同期用のデータを取得する
    /// PDEの場合: 領域の端のデータを返す
    /// ODEの場合: 結合用の状態変数を返す (あるいは空)
    fn get_boundary(&self) -> Vec<f32>;

    /// 設定情報の取得
    fn get_config(&self) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
