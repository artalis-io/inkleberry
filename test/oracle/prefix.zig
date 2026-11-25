// SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
const std=@import("std");
const Engine=@import("engine").Engine;
const Weights=@typeInfo(@typeInfo(@TypeOf(@as(Engine,undefined).weights)).optional.child).pointer.child;
extern "c" fn sq_prefix_test_start([*:0]const u8,u32,u32,u32,u32,u32)c_int;
extern "c" fn sq_ar_test_tokens()[*]const i32;
extern "c" fn sq_prefix_test_check([*]const i32,u32,u32,u32,u32,u64,[*]const i32,u32,u32,u32,u32,u32)c_int;
extern "c" fn sq_prefix_test_sampler(f32,f32,i32)void;
extern "c" fn sq_prefix_test_reset()c_int;
extern "c" fn sq_prefix_test_finish(u32)c_int;
pub fn main(init:std.process.Init.Minimal)u8 {
    const args=init.args.vector;
    if(args.len!=7 and args.len!=8)return 2;
    const edges=args.len==8;
    if(edges and !std.mem.eql(u8,std.mem.span(args[7]),"edges"))return 2;
    const mode=std.fmt.parseInt(u32,std.mem.span(args[2]),10) catch return 2;
    const kv=std.fmt.parseInt(u32,std.mem.span(args[3]),10) catch return 2;
    const graphs=std.fmt.parseInt(u32,std.mem.span(args[4]),10) catch return 2;
    const capacity=std.fmt.parseInt(u32,std.mem.span(args[5]),10) catch return 2;
    const mtp=std.fmt.parseInt(u32,std.mem.span(args[6]),10) catch return 2;
    if(mode>2 or kv>2 or graphs>1 or mtp>1 or capacity<4096 or capacity>131072)return 2;
    var weights:Weights=.{};
    if(!weights.load(std.mem.span(args[1]),@enumFromInt(mode),mtp==1,false,""))return 1;
    defer weights.deinit();
    var engine:Engine=.{};engine.init(&weights,capacity+64,graphs==1,@enumFromInt(kv));
    defer engine.deinit(); engine.enable_prefix_cache();
    if(sq_prefix_test_start(args[1],mode,kv,graphs,capacity,mtp)!=0)return 1;
    const tokens=sq_ar_test_tokens()[0..capacity];
    var output:[64]i32=undefined; var failed=false; var previous_count:u32=0; var edited:[4096]i32=undefined;
    const Case=struct{n:u32,offset:u32=0,hint:u32=128,drafts:u32=0,adaptive:bool=false,seed:u64=42,reset:bool=false,live:bool=false,branch:bool=false,temperature:f32=0.8,p:f32=0.95,k:i32=40};
    const cases=[_]Case{
        .{.n=2049},.{.n=2049,.live=true},.{.n=2049},.{.n=2080},
        .{.n=2080,.branch=true},.{.n=2080},
        .{.n=33,.offset=7},.{.n=2080},.{.n=129},.{.n=2080},
        .{.n=1025,.offset=256},.{.n=2049},
        .{.n=2049,.drafts=if(mtp==1)3 else 0},
        .{.n=2080,.drafts=if(mtp==1)3 else 0},
        .{.n=2080,.drafts=if(mtp==1)7 else 0,.adaptive=mtp==1 and graphs==1,.seed=0},
        .{.n=2080,.drafts=if(mtp==1)7 else 0,.adaptive=mtp==1 and graphs==1,.seed=0},
        .{.n=2049,.reset=true}, .{.n=2049},
        .{.n=capacity-32,.hint=2048},.{.n=capacity-32,.hint=2048},.{.n=capacity-16,.hint=2048},
    };
    const d:u32=if(mtp==1)7 else 0;
    const a=mtp==1 and graphs==1;
    const edge_cases=[_]Case{
        .{.n=2049,.hint=129,.drafts=d,.adaptive=a},
        .{.n=2049,.hint=129,.drafts=d,.adaptive=a,.temperature=0},
        .{.n=1025,.hint=129,.drafts=d,.adaptive=a,.temperature=1.2,.p=0.8,.k=10},
        .{.n=2080,.hint=129,.drafts=d,.adaptive=a},
        .{.n=33,.offset=7,.drafts=d,.adaptive=a},
        .{.n=2080,.hint=129,.drafts=d,.adaptive=a},
        .{.n=33,.offset=7,.drafts=if(mtp==1)3 else 0},
        .{.n=2080,.hint=129,.drafts=d,.adaptive=a},
    };
    const selected=if(edges)edge_cases[0..] else cases[0..];
    for(selected)|c| {
        if(c.reset) {
            // Engine.reset alone does not invalidate Zig's host metadata.
            // Disable its cached matching for this request, then clear it below.
            engine.prefix_cache.session_len=0;engine.prefix_cache.gpu_live_at_pos = -1;
            for(&engine.prefix_cache.checkpoints)|*checkpoint| checkpoint.valid=false;
            if(sq_prefix_test_reset()!=0){failed=true;break;}
        }
        engine.sample_params=.{.temperature=c.temperature,.top_p=c.p,.top_k=c.k,.seed=c.seed};
        sq_prefix_test_sampler(c.temperature,c.p,c.k);
        engine.spec_threshold=if(c.adaptive)0.8 else 0;
        engine.use_prompt_lookup=c.adaptive;
        var prompt:[]const i32=tokens[c.offset..c.offset+c.n];
        if(c.live) {
            @memcpy(edited[0..c.n],prompt);
            @memcpy(edited[c.n..c.n+previous_count],output[0..previous_count]);
            prompt=edited[0..c.n+previous_count];
        } else if(c.branch) {
            @memcpy(edited[0..c.n],prompt);
            edited[1024]=if(edited[1024]==1)2 else 1;
            prompt=edited[0..c.n];
        }
        const stats=engine.generate_with_hint(prompt,c.hint,16,c.drafts,&output,null,null);
        if(sq_prefix_test_check(prompt.ptr,@intCast(prompt.len),c.hint,c.drafts,@intFromBool(c.adaptive),c.seed,&output,
            stats.generated_tokens,stats.cached_tokens,stats.spec_steps,stats.total_drafts,stats.total_accepted)!=0){failed=true;break;}
        previous_count=stats.generated_tokens;
    }
    if(sq_prefix_test_finish(if(failed)0 else 1)!=0)failed=true;
    return if(failed)1 else 0;
}
