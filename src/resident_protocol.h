#pragma once

#include <cstdint>

namespace resident_protocol {
constexpr uint32_t kRecordBytes = 160;
constexpr uint32_t kKeyBytes = 32;
constexpr uint32_t kAddressBytes = 34;
constexpr uint32_t kMaxMatches = 16;
constexpr uint32_t kMetaWords = 5;
constexpr uint32_t kMetaWritePos = 0;
constexpr uint32_t kMetaReadPos = 1;
constexpr uint32_t kMetaOverflow = 2;
constexpr uint32_t kMetaErrors = 3;
constexpr uint32_t kMetaGenerated = 4;
constexpr uint32_t kRecordKey = 0;
constexpr uint32_t kRecordAddress = 32;
constexpr uint32_t kRecordMatchCount = 68;
constexpr uint32_t kRecordMatchIds = 72;
constexpr uint32_t kRecordFlags = 136;
constexpr uint32_t kRecordSequence = 140;
}
