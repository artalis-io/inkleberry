// SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
const std=@import("std");
const Engine=@import("engine").Engine;
const Weights=@typeInfo(@typeInfo(@TypeOf(@as(Engine,undefined).weights)).optional.child).pointer.child;
extern "c" fn sq_adaptive_start([*:0]const u8,u32,u32,u32,u32)c_int;
extern "c" fn sq_adaptive_copy_fixture([*:0]const u8)c_int;
extern "c" fn sq_adaptive_token_count()u32;
extern "c" fn sq_ar_test_tokens()[*]const i32;
extern "c" fn sq_adaptive_check(f32,f32,i32,u64,f32,u32,[*]const i32,u32,u32,u32,u32,u32)c_int;
extern "c" fn sq_adaptive_finish(u32)c_int;
pub fn main(init:std.process.Init.Minimal)u8 {
    const args=init.args.vector;
    if(args.len!=7 and args.len!=8)return 2;
    const mode=std.fmt.parseInt(u32,std.mem.span(args[2]),10) catch return 2;
    const kv=std.fmt.parseInt(u32,std.mem.span(args[3]),10) catch return 2;
    const n=std.fmt.parseInt(u32,std.mem.span(args[4]),10) catch return 2;
    const lookup=std.fmt.parseInt(u32,std.mem.span(args[5]),10) catch return 2;
    const extra=std.fmt.parseInt(u32,std.mem.span(args[6]),10) catch return 2;
    if(mode>2 or kv>2 or lookup>1 or n==0 or n>131072 or (extra!=32 and extra!=64))return 2;
    var weights:Weights=.{};
    if(!weights.load(std.mem.span(args[1]),@enumFromInt(mode),true,false,""))return 1;
    defer weights.deinit();
    var engine:Engine=.{};engine.init(&weights,@max(n+extra,64),true,@enumFromInt(kv));
    defer engine.deinit();
    if(sq_adaptive_start(args[1],mode,kv,n,extra)!=0)return 1;
    const copy=args.len==8 and std.mem.eql(u8,std.mem.span(args[7]),"copy");
    if(copy and sq_adaptive_copy_fixture(args[1])!=0)return 1;
    const token_count=sq_adaptive_token_count();
    const tokens=sq_ar_test_tokens()[0..token_count];var output:[512]i32=undefined;var failed=false;
    const Case=struct{t:f32,p:f32,k:i32,seed:u64,threshold:f32};
    const cases=[_]Case{
        .{.t=0,.p=0.95,.k=0,.seed=0,.threshold=0.8},
        .{.t=0.8,.p=1,.k=0,.seed=0,.threshold=0.2},
        .{.t=0.8,.p=0.95,.k=40,.seed=42,.threshold=1.0},
        .{.t=1.2,.p=0.8,.k=10,.seed=18446744073709551615,.threshold=0.8},
        .{.t=0.8,.p=1,.k=1,.seed=42,.threshold=0.8},
        .{.t=0.00001,.p=0.95,.k=0,.seed=42,.threshold=0.8},
        .{.t=0.8,.p=1,.k=0,.seed=0,.threshold=0.2},
    };
    for(cases) |case| {
        engine.sample_params=.{.temperature=case.t,.top_p=case.p,.top_k=case.k,.seed=case.seed};
        engine.spec_threshold=case.threshold;engine.use_prompt_lookup=lookup==1;
        const stats=engine.generate(tokens,if(copy)256 else 32,7,&output,null,null);
        if(sq_adaptive_check(case.t,case.p,case.k,case.seed,case.threshold,lookup,&output,stats.generated_tokens,
            stats.spec_steps,stats.total_drafts,stats.total_accepted,stats.lookup_steps)!=0){failed=true;break;}
    }
    if(sq_adaptive_finish(if(failed)0 else 1)!=0)failed=true;
    return if(failed)1 else 0;
}
