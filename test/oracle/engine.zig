// SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
// Oracle only: imports the actual pinned engine and weights, unchanged.
const std=@import("std");
const Engine=@import("engine").Engine;
const Weights=@typeInfo(@typeInfo(@TypeOf(@as(Engine,undefined).weights)).optional.child).pointer.child;
extern "c" fn sq_ar_test_start([*:0]const u8,u32,u32,u32,u32,[*:0]const u8)c_int;
extern "c" fn sq_ar_test_tokens()[*]const i32;
extern "c" fn sq_ar_test_logits(usize,[*]const f32,usize)c_int;
extern "c" fn sq_ar_test_generate([*]const i32,usize)c_int;
extern "c" fn sq_ar_test_cancel_prefill()c_int;
extern "c" fn sq_ar_test_reuse()c_int;
extern "c" fn sq_ar_test_finish(u32)c_int;
var chunk:[8*200058]f32=undefined;
fn check(_: *anyopaque,pos:usize,values:[]const f32,n:usize)bool {
    return sq_ar_test_logits(pos,values.ptr,n)==0;
}
pub fn main(init:std.process.Init.Minimal)u8 {
    const args=init.args.vector;
    if(args.len!=7)return 2;
    const mode=std.fmt.parseInt(u32,std.mem.span(args[2]),10) catch return 2;
    const kv=std.fmt.parseInt(u32,std.mem.span(args[3]),10) catch return 2;
    const graph=std.fmt.parseInt(u32,std.mem.span(args[4]),10) catch return 2;
    const n=std.fmt.parseInt(u32,std.mem.span(args[5]),10) catch return 2;
    if(mode>2 or kv>2 or graph>1 or n==0 or n>131072)return 2;
    var weights:Weights=.{};
    if(!weights.load(std.mem.span(args[1]),@enumFromInt(mode),false,false,""))return 1;
    defer weights.deinit();
    var engine:Engine=.{};
    engine.init(&weights,@max(n+64,64),graph==1,@enumFromInt(kv));
    defer engine.deinit();
    if(sq_ar_test_start(args[1],mode,kv,graph,n,args[6])!=0)return 1;
    const tokens=sq_ar_test_tokens()[0..n];
    var ctx:u8=0;
    var failed=!engine.eval_logits_streaming(tokens,&chunk,check,&ctx);
    var output:[64]i32=undefined;
    if(!failed) {
        const stats=engine.generate(tokens,32,0,&output,null,null);
        failed=sq_ar_test_generate(&output,stats.generated_tokens)!=0;
    }
    if(!failed and n<1024)failed=sq_ar_test_reuse()!=0;
    if(!failed and n==8192)failed=sq_ar_test_cancel_prefill()!=0;
    if(sq_ar_test_finish(if(failed)0 else 1)!=0)failed=true;
    return if(failed)1 else 0;
}
