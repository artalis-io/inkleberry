// SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
const std=@import("std");const server=@import("server");const tiger=server.tiger;
pub fn main(init:std.process.Init.Minimal)u8 {
    const args=init.args.vector;if(args.len!=6)return 2;
    var tokenizer:server.tokenizer_mod.Tokenizer=.{};
    if(!tokenizer.load(std.mem.span(args[1])))return 1;defer tokenizer.deinit();
    const fd=tiger.open(args[2],tiger.O_RDONLY);if(fd<0)return 1;defer _=tiger.close(fd);
    const len=tiger.lseek(fd,0,tiger.SEEK_END);if(len<2 or len>32*1024*1024)return 2;
    const mapped=tiger.mmap(null,@intCast(len),tiger.PROT_READ,tiger.MAP_PRIVATE,fd,0);
    if(@intFromPtr(mapped)==tiger.MAP_FAILED_ADDR)return 1;
    defer _=tiger.munmap(mapped,@intCast(len));
    const messages=@as([*]const u8,@ptrCast(mapped))[0..@intCast(len)];
    var buffers=server.ServerBuffers.init();defer buffers.deinit();
    const r=server.encode_chat_template_zero_alloc(&tokenizer,messages,"[]",std.mem.span(args[3]),std.mem.eql(u8,std.mem.span(args[4]),"1"),std.mem.eql(u8,std.mem.span(args[5]),"1"),&buffers);
    if(r.overflow)return 1;
    tiger.print_stdout("PREFIX:{d}\nIDS:",.{r.system_prefix_tokens});
    for(buffers.prompt_tokens[0..r.prompt_len])|id|tiger.print_stdout("{d},",.{id});
    tiger.print_stdout("\n",.{});return 0;
}
