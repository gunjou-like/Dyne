#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"

// Wasm3
#include "wasm3.h"
#include "m3_env.h"

// 生成したヘッダー (Wasmバイナリ)
#include "test_wasm.h"

static const char *TAG = "DYNE_NODE";

// Wasm3が使う内部スタック（WebAssemblyの実行用）
#define WASM_STACK_SIZE 4096

// ★修正点: FreeRTOSタスク自体のスタックサイズを大幅に増やす (32KB)
// これが足りないとParseModuleで落ちます
#define TASK_STACK_SIZE 32768

void wasm_task(void *arg)
{
    ESP_LOGI(TAG, "=== Dyne Node Booting (Task Stack: %d bytes) ===", TASK_STACK_SIZE);

    // 1. Wasm3 環境初期化
    ESP_LOGI(TAG, "Initializing Wasm3...");
    IM3Environment env = m3_NewEnvironment();
    if (!env) { ESP_LOGE(TAG, "m3_NewEnvironment failed"); vTaskDelete(NULL); return; }

    IM3Runtime runtime = m3_NewRuntime(env, WASM_STACK_SIZE, NULL);
    if (!runtime) { ESP_LOGE(TAG, "m3_NewRuntime failed"); vTaskDelete(NULL); return; }

    // 2. モジュールロード
    ESP_LOGI(TAG, "Parsing Wasm Module (%d bytes)...", wasm_blob_len);
    IM3Module module;
    M3Result result = m3_ParseModule(env, &module, wasm_blob, wasm_blob_len);
    if (result) { ESP_LOGE(TAG, "m3_ParseModule: %s", result); vTaskDelete(NULL); return; }

    result = m3_LoadModule(runtime, module);
    if (result) { ESP_LOGE(TAG, "m3_LoadModule: %s", result); vTaskDelete(NULL); return; }

    ESP_LOGI(TAG, "Module Loaded!");

    // 3. 関数を探す
    IM3Function f_alloc, f_step;
    
    result = m3_FindFunction(&f_alloc, runtime, "alloc");
    if (result) { ESP_LOGE(TAG, "Function 'alloc' not found: %s", result); vTaskDelete(NULL); return; }
    
    result = m3_FindFunction(&f_step, runtime, "step");
    if (result) { ESP_LOGE(TAG, "Function 'step' not found: %s", result); vTaskDelete(NULL); return; }

    // 4. 実行テスト
    // (A) メモリ確保: alloc(10)
    const char* i_argv[1] = { "10" }; 
    result = m3_Call(f_alloc, 1, (const void**)i_argv);
    if (result) { ESP_LOGE(TAG, "Call alloc: %s", result); vTaskDelete(NULL); return; }
    
    uint32_t data_ptr_offset = 0;
    const void* ret_ptrs[] = { &data_ptr_offset };
    m3_GetResults(f_alloc, 1, ret_ptrs);

    ESP_LOGI(TAG, "Allocated memory at offset: %ld", data_ptr_offset);

    // (B) データ書き込み
    uint32_t mem_size = 0;
    uint8_t* wasm_mem = m3_GetMemory(runtime, &mem_size, 0);
    float* host_data_ptr = (float*)(wasm_mem + data_ptr_offset);

    printf("Input Data:  ");
    for(int i=0; i<10; i++){
        host_data_ptr[i] = (float)i;
        printf("%.1f ", host_data_ptr[i]);
    }
    printf("\n");

    // (C) 計算実行: step(ptr, len)
    char ptr_str[16];
    char len_str[16];
    sprintf(ptr_str, "%ld", data_ptr_offset);
    sprintf(len_str, "%d", 10);
    const char* step_argv[2] = { ptr_str, len_str };

    ESP_LOGI(TAG, "Running 'step'...");
    result = m3_Call(f_step, 2, (const void**)step_argv);
    if (result) { ESP_LOGE(TAG, "Call step: %s", result); vTaskDelete(NULL); return; }

    // (D) 結果確認
    printf("Output Data: ");
    for(int i=0; i<10; i++){
        printf("%.1f ", host_data_ptr[i]);
    }
    printf("\n");
    
    ESP_LOGI(TAG, "=== Test Complete! ===");

    // タスク終了
    vTaskDelete(NULL);
}

void app_main(void)
{
    // メインの処理を、十分なスタックを持つ別タスクとして起動する
    xTaskCreate(wasm_task, "wasm_task", TASK_STACK_SIZE, NULL, 5, NULL);
}