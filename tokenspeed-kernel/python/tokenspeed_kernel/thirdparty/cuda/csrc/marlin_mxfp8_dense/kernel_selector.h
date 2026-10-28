// MXFP8 W8A16 specializations from the public v0.29 generator.
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 8 &&
    thread_k_blocks == 8 && threads == 256 && m_block_size_8 == true) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 1, 8, 8, true, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 8 &&
    thread_k_blocks == 8 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 1, 8, 8, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 2 && thread_n_blocks == 8 &&
    thread_k_blocks == 8 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 2, 8, 8, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 3 && thread_n_blocks == 8 &&
    thread_k_blocks == 8 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 3, 8, 8, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 4 && thread_n_blocks == 8 &&
    thread_k_blocks == 8 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 4, 8, 8, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 16 &&
    thread_k_blocks == 4 && threads == 256 && m_block_size_8 == true) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 1, 16, 4, true, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 16 &&
    thread_k_blocks == 4 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 1, 16, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 2 && thread_n_blocks == 16 &&
    thread_k_blocks == 4 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 2, 16, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 3 && thread_n_blocks == 16 &&
    thread_k_blocks == 4 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 3, 16, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 4 && thread_n_blocks == 16 &&
    thread_k_blocks == 4 && threads == 256 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  256, 4, 16, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 8 &&
    thread_k_blocks == 4 && threads == 128 && m_block_size_8 == true) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 1, 8, 4, true, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 8 &&
    thread_k_blocks == 4 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 1, 8, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 2 && thread_n_blocks == 8 &&
    thread_k_blocks == 4 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 2, 8, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 3 && thread_n_blocks == 8 &&
    thread_k_blocks == 4 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 3, 8, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 4 && thread_n_blocks == 8 &&
    thread_k_blocks == 4 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 4, 8, 4, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 4 &&
    thread_k_blocks == 8 && threads == 128 && m_block_size_8 == true) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 1, 4, 8, true, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 1 && thread_n_blocks == 4 &&
    thread_k_blocks == 8 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 1, 4, 8, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 2 && thread_n_blocks == 4 &&
    thread_k_blocks == 8 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 2, 4, 8, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 3 && thread_n_blocks == 4 &&
    thread_k_blocks == 8 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 3, 4, 8, false, 4, 2, false>;
}
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu &&
    !has_act_order && !has_zp && !is_zp_float && group_blocks == 2 && stages == 4 &&
    thread_m_blocks == 4 && thread_n_blocks == 4 &&
    thread_k_blocks == 8 && threads == 128 && m_block_size_8 == false) {
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                  vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                  128, 4, 4, 8, false, 4, 2, false>;
}
