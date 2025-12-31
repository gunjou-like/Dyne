static mut BUFFER: [f32; 256] = [0.0; 256];

#[no_mangle]
pub extern "C" fn alloc(_size: usize) -> *mut f32 {
    unsafe { BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn dealloc(_ptr: *mut f32, _size: usize) {
}

#[no_mangle]
// ptr引数は無視して、内部のBUFFERを直接使います
pub extern "C" fn step(_ptr: *mut f32, len: usize) -> *mut f32 {
    unsafe {
        // グローバルバッファの先頭アドレスを取得
        let base_ptr = BUFFER.as_mut_ptr();
        
        // Rawポインタ演算で計算 (Rustのスライスチェックを回避)
        for i in 0..len {
            let offset_ptr = base_ptr.add(i);
            *offset_ptr += 1.0;
        }
        
        base_ptr
    }
}