/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "backend.h"
#include "hipfile.h"

#include <memory>
#include <sys/types.h>

namespace hipFile {

class IBuffer;
class IFile;
class IStream;
enum class IoType;

/// @brief Backend for files registered as hipFileHandleTypeUserspaceFS
///
/// Hands each IO to the read or write callback of the filesystem's hipFileFSOps_t, which moves the data
/// itself (for example by RDMA straight to or from GPU memory). The callback receives the handle the file
/// was registered with and a nullptr hipFileRDMAInfo_t.
class UserspaceFs final : public Backend {
public:
    using Backend::io;
    UserspaceFs()           = default;
    ~UserspaceFs() override = default;

    /// Higher than every kernel-path backend: a userspace filesystem file has no file descriptor of
    /// hipFile's, so no other backend can serve it.
    static constexpr int Score = 200;

    int score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
              hoff_t file_offset, hoff_t buffer_offset) const override;

    void async_io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t *size_p,
                  hoff_t *file_offset_p, hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p,
                  std::shared_ptr<IStream> stream) override;

    /// @brief Enqueue a host function that holds the stream while the filesystem's callback runs
    ///
    /// The callback runs on a thread pool worker rather than in the host function, because a
    /// filesystem may make HIP calls (for example to look up the buffer), which a host function must
    /// not do. No other backend can serve the file, so @p failover is unused.
    void enqueueAsyncIo(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
                        size_t *size_p, hoff_t *file_offset_p, hoff_t *buffer_offset_p,
                        ssize_t *bytes_transferred_p, std::shared_ptr<IStream> stream,
                        std::shared_ptr<AsyncFailoverState> failover) override;

protected:
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset) override;
};

} // namespace hipFile
