/*
 * Copyright (c) 2026 LightSeek Foundation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtPayload.h>
#include <nvtx3/nvToolsExtSemanticsTime.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t start, end;
    uint64_t scope;
    char message[4096];
} ProfileEvent;

typedef struct {
    nvtxDomainHandle_t domain;
    uint64_t schema;
} ProfileContext;

void* ts_nvtx_open(void) {
    ProfileContext* context = calloc(1, sizeof(*context));
    if (!context) return NULL;
    context->domain = nvtxDomainCreateA("TokenSpeed");
    static const nvtxSemanticsTime_t clock = {
        {sizeof(nvtxSemanticsTime_t), NVTX_SEMANTIC_ID_TIME_V1,
         NVTX_TIME_SEMANTIC_VERSION, NULL}, NVTX_TIMESTAMP_TYPE_TOOL_PROVIDED};
    const nvtxPayloadSchemaEntry_t entries[] = {
        {NVTX_PAYLOAD_ENTRY_FLAG_TIMESTAMP | NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN,
         NVTX_PAYLOAD_ENTRY_TYPE_INT64, "start", NULL, 0,
         offsetof(ProfileEvent, start), &clock.header, NULL},
        {NVTX_PAYLOAD_ENTRY_FLAG_TIMESTAMP | NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END,
         NVTX_PAYLOAD_ENTRY_TYPE_INT64, "end", NULL, 0,
         offsetof(ProfileEvent, end), &clock.header, NULL},
        {0, NVTX_PAYLOAD_ENTRY_TYPE_SCOPE_ID, "scope", NULL, 0,
         offsetof(ProfileEvent, scope), NULL, NULL},
        {NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE, NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
         "message", NULL, sizeof(((ProfileEvent*)0)->message),
         offsetof(ProfileEvent, message), NULL, NULL},
    };
    nvtxPayloadSchemaAttr_t attributes = {0};
    attributes.fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
        NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS |
        NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
        NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
        NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE;
    attributes.type = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC;
    attributes.flags = NVTX_PAYLOAD_SCHEMA_FLAG_RANGE_PUSHPOP;
    attributes.entries = entries;
    attributes.numEntries = sizeof(entries) / sizeof(entries[0]);
    /* Nsight needs the full size to decode deferred events. */
    attributes.payloadStaticSize = sizeof(ProfileEvent);
    context->schema = nvtxPayloadSchemaRegister(context->domain, &attributes);
    if (!context->schema || !nvtxDomainIsEnabled(context->domain)) {
        nvtxDomainDestroy(context->domain);
        free(context);
        return NULL;
    }
    return context;
}

int64_t ts_nvtx_timestamp(void) { return nvtxTimestampGet(); }

int ts_nvtx_submit(void* handle, int64_t start, int64_t end, const char* label) {
    ProfileContext* context = handle;
    if (!context || !label || end < start || strlen(label) >= 4096) return -1;
    ProfileEvent event = {0};
    event.start = start;
    event.end = end;
    event.scope = NVTX_SCOPE_CURRENT_SW_THREAD;
    memcpy(event.message, label, strlen(label) + 1);
    const nvtxPayloadData_t payload = {context->schema, sizeof(event), &event};
    nvtxEventSubmit(context->domain, &payload, 1);
    return 0;
}

void ts_nvtx_close(void* handle) {
    ProfileContext* context = handle;
    if (context) {
        nvtxDomainDestroy(context->domain);
        free(context);
    }
}
