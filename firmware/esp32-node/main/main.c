#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"

// Wasm3
#include "wasm3.h"
#include "m3_env.h"

// 物理エンジンのヘッダー
#include "dyne_wasm.h"

static const char *TAG = "DYNE_PHYSICS";

#define WASM_STACK_SIZE 4096
#define TASK_STACK_SIZE 32768

// コンソールに波の高さをグラフ表示するヘルパー関数
void print_wave_ascii(float* data, int size) {
    char buf[256];
    int buf_idx = 0;
    
    // 全データを表示すると多すぎるので、4つ飛ばしで間引き表示
    for (int i = 0; i < size; i += 4) {
        if (buf_idx >= 250) break; // バッファオーバーラン防止

        float val = data[i];
        char c = '_';
        if (val > 0.8) c = '#';
        else if (val > 0.6) c = '=';
        else if (val > 0.4) c = '-';
        else if (val > 0.2) c = '.';
        
        buf[buf_idx++] = c;
    }
    buf[buf_idx] = '\0';
    
    // 中央付近の生データも少し表示
    printf("[%s] Center: %.3f\n", buf, data[size/2]);
}

void wasm_task(void *arg)
{
    ESP_LOGI(TAG, "=== Dyne Physics Engine Starting ===");

    // 1. 初期化
    IM3Environment env = m3_NewEnvironment();
    IM3Runtime runtime = m3_NewRuntime(env, WASM_STACK_SIZE, NULL);
    IM3Module module;
    
    M3Result result = m3_ParseModule(env, &module, dyne_wasm, dyne_wasm_len);
    if (result) { ESP_LOGE(TAG, "Parse: %s", result); vTaskDelete(NULL); return; }

    result = m3_LoadModule(runtime, module);
    if (result) { ESP_LOGE(TAG, "Load: %s", result); vTaskDelete(NULL); return; }

    ESP_LOGI(TAG, "Module Loaded! (%d bytes)", dyne_wasm_len);

    // 2. 関数検索
    IM3Function f_init, f_step, f_get_size;
    m3_FindFunction(&f_init, runtime, "init");
    m3_FindFunction(&f_step, runtime, "step");
    m3_FindFunction(&f_get_size, runtime, "get_size");

    if (!f_init || !f_step || !f_get_size) {
        ESP_LOGE(TAG, "Functions not found!");
        vTaskDelete(NULL);
        return;
    }

    // 3. サイズ取得 (ここを修正しました)
    uint64_t grid_size_u64 = 0;
    m3_Call(f_get_size, 0, NULL);
    
    // ★修正点: ポインタの配列に入れてから渡す
    const void* size_ret_ptrs[] = { &grid_size_u64 };
    m3_GetResults(f_get_size, 1, size_ret_ptrs);
    
    int grid_size = (int)grid_size_u64;
    ESP_LOGI(TAG, "Grid Size: %d", grid_size);

    // 4. 初期化 (init実行) -> 初期状態のポインタ取得
    uint32_t data_offset = 0;
    m3_Call(f_init, 0, NULL);
    
    // ここも同様に配列経由で渡す (前回からここは合っていました)
    const void* data_ret_ptrs[] = { &data_offset };
    m3_GetResults(f_init, 1, data_ret_ptrs);
    
    ESP_LOGI(TAG, "Data Buffer Offset: %ld", data_offset);

    // Wasmメモリへのアクセスポインタ取得
    uint32_t mem_size = 0;
    uint8_t* base_mem = m3_GetMemory(runtime, &mem_size, 0);
    
    // オフセットがメモリ範囲内かチェック
    if (data_offset >= mem_size) {
        ESP_LOGE(TAG, "Invalid offset: %ld (MemSize: %ld)", data_offset, mem_size);
        vTaskDelete(NULL);
        return;
    }

    float* wave_data = (float*)(base_mem + data_offset);

    ESP_LOGI(TAG, "Simulation Start! (Gaussian Pulse)");

    // 5. メインループ (50ステップ実行)
    for (int t = 0; t < 50; t++) {
        // Rustのステップ関数を実行
        result = m3_Call(f_step, 0, NULL);
        if (result) {
            ESP_LOGE(TAG, "Step Error: %s", result);
            break;
        }

        // 結果を表示
        print_wave_ascii(wave_data, grid_size);
        
        // 少しウェイトを入れて見やすくする (50ms)
        vTaskDelay(50 / portTICK_PERIOD_MS);
    }

    ESP_LOGI(TAG, "=== Simulation Complete ===");
    vTaskDelete(NULL);
}

void app_main(void)
{
    xTaskCreate(wasm_task, "wasm_task", TASK_STACK_SIZE, NULL, 5, NULL);
}