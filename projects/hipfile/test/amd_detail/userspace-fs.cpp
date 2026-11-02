/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend.h"
#include "backend/userspace-fs.h"
#include "hip.h"
#include "hipfile.h"
#include "hipfile-warnings.h"
#include "io.h"
#include "masyncmonitor.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mhip.h"
#include "mstream.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using namespace hipFile;
using namespace testing;
using namespace std;

// Put tests inside the macros to suppress the global constructor warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

namespace {

// What the last callback received, and what the next one returns. Each test runs in its own process.
struct FsCall {
    void              *handle;
    const char        *buf;
    size_t             size;
    hoff_t             offset;
    hipFileRDMAInfo_t *rdma_info;
    bool               write;
};

FsCall          last_call{};
ssize_t         next_result{0};
std::thread::id fs_thread{};

ssize_t
fsRead(void *handle, char *buf, size_t size, hoff_t offset, hipFileRDMAInfo_t *rdma_info)
{
    last_call = {handle, buf, size, offset, rdma_info, false};
    fs_thread = std::this_thread::get_id();
    return next_result;
}

ssize_t
fsWrite(void *handle, const char *buf, size_t size, hoff_t offset, hipFileRDMAInfo_t *rdma_info)
{
    last_call = {handle, buf, size, offset, rdma_info, true};
    fs_thread = std::this_thread::get_id();
    return next_result;
}

const hipFileFSOps_t fs_ops{nullptr, nullptr, nullptr, fsRead, fsWrite};
const hipFileFSOps_t read_only_ops{nullptr, nullptr, nullptr, fsRead, nullptr};

} // namespace

struct UserspaceFsTest : public testing::Test {
    int                             fs_file{};
    const size_t                    io_size{2048};
    const hoff_t                    file_offset{4096};
    const hoff_t                    buffer_offset{1024};
    array<char, 8192>               data{};
    shared_ptr<StrictMock<MFile>>   mfile{make_shared<StrictMock<MFile>>()};
    shared_ptr<StrictMock<MBuffer>> mbuffer{make_shared<StrictMock<MBuffer>>()};

    void expectIo(const hipFileFSOps_t *ops)
    {
        EXPECT_CALL(*mfile, fsOps).WillRepeatedly(Return(ops));
        EXPECT_CALL(*mfile, fsHandle).WillRepeatedly(Return(&fs_file));
        EXPECT_CALL(*mbuffer, getBuffer).WillRepeatedly(Return(data.data()));
        EXPECT_CALL(*mbuffer, getLength).WillRepeatedly(Return(data.size()));
    }
};

TEST_F(UserspaceFsTest, ScoreAcceptsFileWithFsOps)
{
    EXPECT_CALL(*mfile, fsOps).WillOnce(Return(&fs_ops));
    ASSERT_EQ(UserspaceFs().score(mfile, mbuffer, io_size, file_offset, buffer_offset), UserspaceFs::Score);
}

TEST_F(UserspaceFsTest, ScoreRejectsKernelFile)
{
    EXPECT_CALL(*mfile, fsOps).WillOnce(Return(nullptr));
    ASSERT_EQ(UserspaceFs().score(mfile, mbuffer, io_size, file_offset, buffer_offset), -1);
}

TEST_F(UserspaceFsTest, ScoreRejectsFsOpsWithoutBothCallbacks)
{
    EXPECT_CALL(*mfile, fsOps).WillOnce(Return(&read_only_ops));
    ASSERT_EQ(UserspaceFs().score(mfile, mbuffer, io_size, file_offset, buffer_offset), -1);
}

TEST_F(UserspaceFsTest, ReadPassesHandleBufferRegionAndNullRdmaInfo)
{
    expectIo(&fs_ops);
    next_result = static_cast<ssize_t>(io_size);

    ASSERT_EQ(UserspaceFs().io(IoType::Read, mfile, mbuffer, io_size, file_offset, buffer_offset),
              static_cast<ssize_t>(io_size));
    EXPECT_FALSE(last_call.write);
    EXPECT_EQ(last_call.handle, &fs_file);
    EXPECT_EQ(last_call.buf, data.data() + buffer_offset);
    EXPECT_EQ(last_call.size, io_size);
    EXPECT_EQ(last_call.offset, file_offset);
    EXPECT_EQ(last_call.rdma_info, nullptr);
}

TEST_F(UserspaceFsTest, WriteCallsWriteCallback)
{
    expectIo(&fs_ops);
    next_result = 100;

    ASSERT_EQ(UserspaceFs().io(IoType::Write, mfile, mbuffer, io_size, file_offset, buffer_offset), 100);
    EXPECT_TRUE(last_call.write);
    EXPECT_EQ(last_call.buf, data.data() + buffer_offset);
}

TEST_F(UserspaceFsTest, NegativeResultThrowsItsErrno)
{
    expectIo(&fs_ops);
    next_result = -ENOSPC;

    try {
        UserspaceFs().io(IoType::Write, mfile, mbuffer, io_size, file_offset, buffer_offset);
        FAIL() << "expected std::system_error";
    }
    catch (const std::system_error &e) {
        EXPECT_EQ(e.code().value(), ENOSPC);
    }
}

TEST_F(UserspaceFsTest, ResultLargerThanRequestThrowsEio)
{
    expectIo(&fs_ops);
    next_result = static_cast<ssize_t>(io_size) + 1;

    try {
        UserspaceFs().io(IoType::Read, mfile, mbuffer, io_size, file_offset, buffer_offset);
        FAIL() << "expected std::system_error";
    }
    catch (const std::system_error &e) {
        EXPECT_EQ(e.code().value(), EIO);
    }
}

TEST_F(UserspaceFsTest, IoThrowsIfOpCouldOverrunBuffer)
{
    expectIo(&fs_ops);
    ASSERT_THROW(UserspaceFs().io(IoType::Read, mfile, mbuffer, data.size(), file_offset, buffer_offset),
                 std::invalid_argument);
}

TEST_F(UserspaceFsTest, IoThrowsOnNegativeFileOffset)
{
    expectIo(&fs_ops);
    ASSERT_THROW(UserspaceFs().io(IoType::Read, mfile, mbuffer, io_size, -1, buffer_offset),
                 std::invalid_argument);
}

TEST_F(UserspaceFsTest, IoRejectedWithoutBothCallbacks)
{
    expectIo(&read_only_ops);
    ASSERT_THROW(UserspaceFs().io(IoType::Read, mfile, mbuffer, io_size, file_offset, buffer_offset),
                 BackendDisabled);
}

// The stream is mocked: host functions are recorded at enqueue and run by runStream().
struct UserspaceFsAsyncTest : public UserspaceFsTest, public WithParamInterface<IoType> {
    StrictMock<MHip>                  mhip;
    StrictMock<MAsyncMonitor>         mmonitor;
    shared_ptr<StrictMock<MStream>>   mstream{make_shared<StrictMock<MStream>>()};
    shared_ptr<AsyncOp>               op;
    vector<pair<hipHostFn_t, void *>> host_fns;
    size_t                            size{io_size};
    hoff_t                            foff{file_offset};
    hoff_t                            boff{buffer_offset};
    ssize_t                           nbytes{-99};

    void SetUp() override
    {
        last_call   = {};
        next_result = static_cast<ssize_t>(io_size);
        expectIo(&fs_ops);
        fixed(false);
    }

    void fixed(bool params_fixed)
    {
        EXPECT_CALL(*mstream, fixedIOSize).WillRepeatedly(Return(params_fixed));
        EXPECT_CALL(*mstream, fixedFileOffset).WillRepeatedly(Return(params_fixed));
        EXPECT_CALL(*mstream, fixedBufferOffset).WillRepeatedly(Return(params_fixed));
    }

    void expectEnqueue()
    {
        EXPECT_CALL(*mstream, getLock);
        EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
        EXPECT_CALL(mmonitor, addOp).WillOnce(SaveArg<0>(&op));
        EXPECT_CALL(mhip, hipLaunchHostFunc).WillOnce([this](hipStream_t, hipHostFn_t fn, void *arg) {
            host_fns.emplace_back(fn, arg);
        });
    }

    void enqueue()
    {
        UserspaceFs().async_io(GetParam(), mfile, mbuffer, &size, &foff, &boff, &nbytes, mstream);
    }

    void runStream()
    {
        EXPECT_CALL(mmonitor, completeOp(op.get()));
        for (auto [fn, arg] : host_fns) {
            fn(arg);
        }
    }
};

TEST_P(UserspaceFsAsyncTest, IoRunsOnAWorkerWhenTheStreamReachesIt)
{
    expectEnqueue();
    enqueue();
    ASSERT_EQ(host_fns.size(), 1u);
    ASSERT_NE(host_fns[0].first, &async_io_cleanup);
    ASSERT_EQ(last_call.handle, nullptr);
    ASSERT_EQ(nbytes, 0);

    runStream();
    ASSERT_EQ(last_call.handle, &fs_file);
    ASSERT_EQ(last_call.buf, data.data() + buffer_offset);
    ASSERT_EQ(last_call.size, io_size);
    ASSERT_EQ(last_call.offset, file_offset);
    ASSERT_EQ(last_call.rdma_info, nullptr);
    ASSERT_EQ(last_call.write, GetParam() == IoType::Write);
    ASSERT_NE(fs_thread, std::this_thread::get_id());
    ASSERT_EQ(nbytes, static_cast<ssize_t>(io_size));
}

TEST_P(UserspaceFsAsyncTest, UnfixedParametersAreReadWhenTheStreamReachesIt)
{
    expectEnqueue();
    enqueue();
    size        = 512;
    foff        = 8192;
    boff        = 0;
    next_result = 512;

    runStream();
    ASSERT_EQ(last_call.buf, data.data());
    ASSERT_EQ(last_call.size, 512u);
    ASSERT_EQ(last_call.offset, 8192);
    ASSERT_EQ(nbytes, 512);
}

TEST_P(UserspaceFsAsyncTest, FixedParametersAreReadAtEnqueue)
{
    fixed(true);
    expectEnqueue();
    enqueue();
    size = 512;
    foff = 8192;
    boff = 0;

    runStream();
    ASSERT_EQ(last_call.buf, data.data() + buffer_offset);
    ASSERT_EQ(last_call.size, io_size);
    ASSERT_EQ(last_call.offset, file_offset);
}

TEST_P(UserspaceFsAsyncTest, FilesystemErrorReportsFailure)
{
    next_result = -EIO;
    expectEnqueue();
    enqueue();

    runStream();
    ASSERT_EQ(nbytes, -1);
}

TEST_P(UserspaceFsAsyncTest, RegionOverrunWhenTheStreamReachesItIsInvalidValue)
{
    expectEnqueue();
    enqueue();
    size = data.size();

    runStream();
    ASSERT_EQ(last_call.handle, nullptr);
    ASSERT_EQ(nbytes, -hipFileInvalidValue);
}

TEST_P(UserspaceFsAsyncTest, InvalidRegionAtEnqueueThrows)
{
    boff = static_cast<hoff_t>(data.size());
    ASSERT_THROW(enqueue(), std::invalid_argument);
}

TEST_P(UserspaceFsAsyncTest, ZeroSizeEnqueuesNothing)
{
    size = 0;
    enqueue();
    ASSERT_EQ(nbytes, 0);
}

TEST_P(UserspaceFsAsyncTest, FailedEnqueueStillEnqueuesCleanup)
{
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    EXPECT_CALL(mmonitor, addOp).WillOnce(SaveArg<0>(&op));
    EXPECT_CALL(mhip, hipLaunchHostFunc)
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)))
        .WillOnce([this](hipStream_t, hipHostFn_t fn, void *arg) { host_fns.emplace_back(fn, arg); });

    ASSERT_THROW(enqueue(), Hip::RuntimeError);
    ASSERT_EQ(host_fns.size(), 1u);
    ASSERT_EQ(host_fns[0].first, &async_io_cleanup);
    runStream();
    ASSERT_EQ(last_call.handle, nullptr);
}

TEST_P(UserspaceFsAsyncTest, WithoutBothCallbacksThrows)
{
    expectIo(&read_only_ops);
    ASSERT_THROW(enqueue(), BackendDisabled);
}

INSTANTIATE_TEST_SUITE_P(UserspaceFsAsync, UserspaceFsAsyncTest, Values(IoType::Read, IoType::Write));

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
