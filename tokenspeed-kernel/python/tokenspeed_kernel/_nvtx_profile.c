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
#include <stdlib.h>

typedef struct {
    nvtxDomainHandle_t domain;
} ProfileContext;

void* ts_nvtx_open(void) {
    ProfileContext* context = calloc(1, sizeof(*context));
    if (!context) return NULL;
    context->domain = nvtxDomainCreateA("TokenSpeed");
    if (!nvtxDomainIsEnabled(context->domain)) {
        nvtxDomainDestroy(context->domain);
        free(context);
        return NULL;
    }
    return context;
}

int ts_nvtx_push(void* handle, const char* label, uint64_t scope_id) {
    ProfileContext* context = handle;
    if (!context || !label) return -1;
    nvtxEventAttributes_t event = {0};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = label;
    event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    event.payload.ullValue = scope_id;
    /* Tool return values are not portable, even when the range is recorded. */
    nvtxDomainRangePushEx(context->domain, &event);
    return 0;
}

int ts_nvtx_pop(void* handle) {
    ProfileContext* context = handle;
    if (!context) return -1;
    nvtxDomainRangePop(context->domain);
    return 0;
}

void ts_nvtx_close(void* handle) {
    ProfileContext* context = handle;
    if (context) {
        nvtxDomainDestroy(context->domain);
        free(context);
    }
}
