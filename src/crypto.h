#pragma once
#include <array>
#include <string>
#include <vector>

// base58 字母表（Tron / Bitcoin）
extern const char kBase58[59];

std::string bytesToHexUpper(const unsigned char* data, size_t len);

// 由未压缩公钥的 64 字节 (X||Y，不含 0x04 前缀) 计算 Tron Base58Check 地址
std::string tronAddressFromPubXY(const unsigned char pubXY[64]);

// The 25 bytes before Base58Check text encoding: 0x41 || 20-byte account || 4-byte checksum.
// Useful as a CPU reference for staged GPU address pipelines.
void tronFullFromPubXY(const unsigned char pubXY[64], unsigned char full[25]);
