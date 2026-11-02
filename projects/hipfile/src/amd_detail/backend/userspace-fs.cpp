/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend/userspace-fs.h"

#include "async.h"
#include "backend.h"
#include "buffer.h"
#include "context.h"
#include "file.h"
#include "hip.h"
#include "hipfile.h"
#include "io.h"
#include "stream.h"
#include "sys.h"
#include "thread-pool.h"
#include "util.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <syslog.h>
#include <system_error>

using namespace hipFile;

int
UserspaceFs::score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
                   hoff_t file_offset, hoff_t buffer_offset) const
{
    (void)buffer;
    (void)size;
    (void)file_offset;
    (void)buffer_offset;

    const hipFileFSOps_t *ops{file->fsOps()};
    if (ops == nullptr || ops->read == nullptr || ops->write == nullptr) {
        return -1;
    }
    return Score;
}

namespace {

ssize_t
fsIo(IoType type, IFile &file, IBuffer &buffer, size_t size, hoff_t file_offset, hoff_t buffer_offset)
{
    const hipFileFSOps_t *ops{file.fsOps()};
    char                 *buf{static_cast<char *>(buffer.getBuffer()) + buffer_offset};
    void                 *handle{file.fsHandle()};
    ssize_t               nbytes{type == IoType::Read ? ops->read(handle, buf, size, file_offset, nullptr)
                                                      : ops->write(handle, buf, size, file_offset, nullptr)};

    if (nbytes < 0) {
        throw std::system_error(static_cast<int>(-nbytes), std::generic_category());
    }
    // A filesystem that claims more than it was asked for has a bug; don't pass that on.
    if (static_cast<size_t>(nbytes) > size) {
        throw std::system_error(EIO, std::generic_category());
    }
    return nbytes;
}

// Reads the parameters the stream may have left unbound until now, the way the fallback does.
ssize_t
asyncFsIo(AsyncOp &op) noexcept
try {
    const size_t size{std::min(*get_variant_ptr(op.size), hipFile::getMaxRwCount())};
    const hoff_t file_offset{*get_variant_ptr(op.file_offset)};
    const hoff_t buffer_offset{*get_variant_ptr(op.buffer_offset)};

    if (!paramsValid(op.buffer, size, file_offset, buffer_offset)) {
        return -hipFileInvalidValue;
    }
    return fsIo(op.io_type, *op.file, *op.buffer, size, file_offset, buffer_offset);
}
catch (const std::system_error &) {
    return -1;
}
catch (...) {
    return -hipFileInternalError;
}

} // namespace

extern "C" {
static void
userspace_fs_async_io(void *userargs)
{
    auto op = static_cast<AsyncOp *>(userargs);
    try {
        auto tasks = Context<IThreadPool>::get()->makeTaskGroup();
        tasks->run([op] { op->bytes_transferred_internal = asyncFsIo(*op); });
        tasks->wait();
    }
    catch (...) {
        op->bytes_transferred_internal = -hipFileInternalError;
    }
    // The IO is never split, so it publishes its result here rather than in a host function of its own.
    async_io_cleanup(op);
}
}

ssize_t
UserspaceFs::_io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                      hoff_t file_offset, hoff_t buffer_offset)
{
    const hipFileFSOps_t *ops{file->fsOps()};
    if (ops == nullptr || ops->read == nullptr || ops->write == nullptr) {
        throw BackendDisabled();
    }

    size = std::min(size, hipFile::getMaxRwCount());

    if (!paramsValid(buffer, size, file_offset, buffer_offset)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

    return fsIo(type, *file, *buffer, size, file_offset, buffer_offset);
}

void
UserspaceFs::async_io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
                      size_t *size_p, hoff_t *file_offset_p, hoff_t *buffer_offset_p,
                      ssize_t *bytes_transferred_p, std::shared_ptr<IStream> stream)
{
    enqueueAsyncIo(type, std::move(file), std::move(buffer), size_p, file_offset_p, buffer_offset_p,
                   bytes_transferred_p, std::move(stream), nullptr);
}

void
UserspaceFs::enqueueAsyncIo(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
                            size_t *size_p, hoff_t *file_offset_p, hoff_t *buffer_offset_p,
                            ssize_t *bytes_transferred_p, std::shared_ptr<IStream> stream,
                            std::shared_ptr<AsyncFailoverState>)
{
    const hipFileFSOps_t *ops{file->fsOps()};
    if (ops == nullptr || ops->read == nullptr || ops->write == nullptr) {
        throw BackendDisabled();
    }

    if (!paramsValid(buffer, std::min(*size_p, hipFile::getMaxRwCount()), *file_offset_p, *buffer_offset_p)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

    *bytes_transferred_p = 0;
    if (*size_p == 0) {
        return;
    }

    auto op = std::make_shared<AsyncOp>(type, std::move(file), std::move(buffer), stream, size_p,
                                        file_offset_p, buffer_offset_p, bytes_transferred_p);
    Context<AsyncMonitor>::get()->addOp(op);

    try {
        auto stream_lock = stream->getLock();
        Context<Hip>::get()->hipLaunchHostFunc(stream->getHipStream(), userspace_fs_async_io, op.get());
    }
    catch (...) {
        // If something threw, still try to enqueue cleanup function
        try {
            Context<Hip>::get()->hipLaunchHostFunc(stream->getHipStream(), async_io_cleanup, op.get());
        }
        catch (...) {
            Context<Sys>::get()->syslog(LOG_CRIT,
                                        "Unable to enqueue async cleanup function. This will leak memory.");
        }
        throw;
    }
}
