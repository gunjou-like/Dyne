/// Dyneシステム上で動作する物理エンジンの共通インターフェース
pub trait DyneEngine {
    /// 1ステップ時間を進める
    /// 
    /// # Arguments
    /// * `input` - 現在の境界条件や入力データ
    /// 
    /// # Returns
    /// * 次のタイムステップの状態データ
    fn step(&mut self, input: &[f32]) -> Vec<f32>;

    /// 設定されている物理パラメータ（グリッド幅など）を取得する
    /// (デバッグや可視化用)
    fn get_config(&self) -> String;
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
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
