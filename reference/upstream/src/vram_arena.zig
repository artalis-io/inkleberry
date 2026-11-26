// TigerStyle single-allocation VRAM bump arena.
// Replaces ~600 individual cudaMalloc calls (weights) and ~450 individual cudaMalloc calls (runtime)
// with ONE contiguous cudaMalloc per lifetime, paired alignment assertions, and exact byte accounting.

const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const tiger = @import("tiger.zig");

pub const VRAMArena = struct {
    base_ptr: ?[*]u8 = null,
    capacity_bytes: u64 = 0,
    offset_bytes: u64 = 0,
    allocation_count: u32 = 0,

    pub fn init(capacity_bytes: u64) VRAMArena {
        tiger.assert(capacity_bytes > 0);
        const aligned_capacity: u64 = tiger.align_up_u64(capacity_bytes, constants.vram_alignment_bytes);
        var raw_ptr: ?*anyopaque = null;
        const rc: i32 = cuda.squid_cuda_malloc(&raw_ptr, @intCast(aligned_capacity));
        tiger.assert_cuda(rc);
        tiger.assert(raw_ptr != null);

        const base: [*]u8 = @ptrCast(raw_ptr.?);
        tiger.assert(@intFromPtr(base) % constants.vram_alignment_bytes == 0);
        return .{
            .base_ptr = base,
            .capacity_bytes = aligned_capacity,
            .offset_bytes = 0,
            .allocation_count = 0,
        };
    }

    pub fn deinit(self: *VRAMArena) void {
        tiger.assert(self.offset_bytes <= self.capacity_bytes);
        if (self.base_ptr) |base| {
            const rc: i32 = cuda.squid_cuda_free(@ptrCast(base));
            tiger.assert_cuda(rc);
        }
        self.base_ptr = null;
        self.capacity_bytes = 0;
        self.offset_bytes = 0;
        self.allocation_count = 0;
    }

    /// Bump-allocate `bytes` aligned to 256 bytes from the preallocated contiguous VRAM block.
    pub fn alloc_raw(self: *VRAMArena, bytes: u64) *anyopaque {
        tiger.assert(self.base_ptr != null);
        tiger.assert(bytes > 0);
        const aligned_bytes: u64 = tiger.align_up_u64(bytes, constants.vram_alignment_bytes);
        tiger.assert(aligned_bytes <= self.capacity_bytes - self.offset_bytes);

        const start_offset: u64 = self.offset_bytes;
        const ptr: [*]u8 = self.base_ptr.? + @as(usize, @intCast(start_offset));
        self.offset_bytes += aligned_bytes;
        self.allocation_count += 1;

        tiger.assert(@intFromPtr(ptr) % constants.vram_alignment_bytes == 0);
        tiger.assert(self.offset_bytes <= self.capacity_bytes);
        return @ptrCast(ptr);
    }

    /// Bump-allocate `count` elements of `T` as a many-item pointer.
    pub fn alloc_many(self: *VRAMArena, comptime T: type, count: u64) [*]T {
        tiger.assert(count > 0);
        const raw: *anyopaque = self.alloc_raw(count * @sizeOf(T));
        return @ptrCast(@alignCast(raw));
    }

    /// Bump-allocate a single scalar `T` in VRAM.
    pub fn alloc_one(self: *VRAMArena, comptime T: type) *T {
        const raw: *anyopaque = self.alloc_raw(@sizeOf(T));
        return @ptrCast(@alignCast(raw));
    }

    /// Paired post-condition assertion verifying Pass 1 (size calculation) == Pass 2 (actual bump usage).
    pub fn assert_exact_fill(self: *const VRAMArena) void {
        tiger.assert(self.base_ptr != null);
        tiger.assert(self.offset_bytes == self.capacity_bytes);
    }
};
