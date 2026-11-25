// SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
// Test harness only: original weights.zig and its dependencies stay unchanged.
const std = @import("std");
const weights_mod = @import("weights");
extern "c" fn sq_test_start([*:0]const u8, u32, u32, u64, u64, ?*const anyopaque) c_int;
extern "c" fn sq_test_view([*:0]const u8, u32, u32, u32, u32, u32, u64, u64, u64, u64, f32) c_int;
extern "c" fn sq_test_finish(u32, u32, u32) c_int;
fn labelZ(buffer: []u8, name: []const u8) error{NoSpaceLeft}![:0]const u8 {
    if (name.len >= buffer.len) return error.NoSpaceLeft;
    @memcpy(buffer[0..name.len],name);
    buffer[name.len]=0;
    return buffer[0..name.len :0];
}
fn offset(value: anytype, base: usize, capacity: usize) u64 {
    if (value) |p| {
        const address = @intFromPtr(p);
        if (address < base or address - base >= capacity) return std.math.maxInt(u64)-1;
        return address-base;
    }
    return std.math.maxInt(u64);
}
fn walk(value: anytype, name: []const u8, base: usize, capacity: usize) bool {
    const T = @TypeOf(value);
    var buffer: [192]u8 = undefined;
    switch (@typeInfo(T)) {
        .@"struct" => {
            if (@hasField(T,"global_scale") and @hasField(T,"row_scales")) {
                if (value.data == null) return true;
                const label = labelZ(&buffer,name) catch return false;
                return sq_test_view(label,1,value.kind,@intCast(value.rows),@intCast(value.cols),0,
                    offset(value.data,base,capacity),offset(value.scales,base,capacity),offset(value.row_scales,base,capacity),std.math.maxInt(u64),value.global_scale)==0;
            } else if (@hasField(T,"n_experts") and @hasField(T,"expert_scale")) {
                if (value.data == null) return true;
                const label = labelZ(&buffer,name) catch return false;
                return sq_test_view(label,2,if(value.is_nvfp4!=0) 2 else 1,@intCast(value.rows),@intCast(value.cols),@intCast(value.n_experts),
                    offset(value.data,base,capacity),offset(value.scales,base,capacity),std.math.maxInt(u64),offset(value.expert_scale,base,capacity),1.0)==0;
            } else {
                inline for (@typeInfo(T).@"struct".field_names) |field| {
                    if (!(name.len==0 and (std.mem.eql(u8,field,"arena") or std.mem.eql(u8,field,"ablation")))) {
                        const child = std.fmt.bufPrint(&buffer,"{s}{s}{s}",.{name,if(name.len==0) "" else ".",field}) catch return false;
                        if (!walk(@field(value,field),child,base,capacity)) return false;
                    }
                }
            }
        },
        .array => { for (value,0..) |child,i| {
            const label = std.fmt.bufPrint(&buffer,"{s}.{d}",.{name,i}) catch return false;
            if(!walk(child,label,base,capacity))return false;
        } },
        .optional => {
            if(value) |p| {
                if (@typeInfo(@TypeOf(p))==.pointer) {
                    const label = labelZ(&buffer,name) catch return false;
                    return sq_test_view(label,0,0,0,0,0,@intFromPtr(p)-base,std.math.maxInt(u64),std.math.maxInt(u64),std.math.maxInt(u64),1.0)==0;
                }
            }
        },
        else => {},
    }
    return true;
}
pub fn main(init: std.process.Init.Minimal) u8 {
    const args=init.args.vector;
    if(args.len!=4)return 2;
    const mode=std.fmt.parseInt(u32,std.mem.span(args[2]),10) catch return 2;
    const mtp=std.fmt.parseInt(u32,std.mem.span(args[3]),10) catch return 2;
    if(mode>2 or mtp>1)return 2;
    var original: weights_mod.ModelWeights=.{};
    // Explicit full vocabulary and default MTP precision selected by test env.
    if(!original.load(std.mem.span(args[1]),@enumFromInt(mode),mtp==1,false,""))return 1;
    defer original.deinit();
    var result: u8=0;
    if(sq_test_start(args[1],mode,mtp,original.arena.capacity_bytes,original.arena.allocation_count,@ptrCast(original.arena.base_ptr.?))!=0) result=1
    else if(!walk(original,"",@intFromPtr(original.arena.base_ptr.?),@intCast(original.arena.capacity_bytes))) result=1;
    if(sq_test_finish(mode,mtp,if(result==0) 1 else 0)!=0)result=1;
    return result;
}
