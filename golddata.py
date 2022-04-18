snap_gold = [
    [0x000000, 0x001000, "bin", "boot"],
    [0x001000, 0x033BD4, "asm", "app", 0x80000400],
    [0x033BD4, 0x046270, "bin"],
    [0x046270, 0x057574, "asm", "app_render", 0x8009A8C0],
    [0x057574, 0x05BF20, "bin"],
    [0x05BF20, 0x05EF80, "asm", "more_funcs", 0x800BF080],
    [0x05EF80, 0x05F050, "bin"],
    [0x05F050, 0x064280, "asm", "level_low", 0x800E18A0],
    [0x064280, 0x4F0610, "bin"],
    [0x4F0610, 0x506FB0, "asm", "app_level", 0x80350200],
    [0x506FB0, 0x55C110, "bin"],
    [0x55C110, 0x563E80, "asm", "beach_code", 0x802C40A0],
    [0x563E80, 0x5DF5D0, "bin"],
    [0x5DF5D0, 0x5EAFD0, "asm", "tunnel_code", 0x802E2500],
    [0x5EAFD0, 0x6401B0, "bin"],
    [0x6401B0, 0x6485F4, "asm", "cave_code", 0x802BDD00],
    [0x6485F4, 0x6C05E0, "bin"],
    [0x6C05E0, 0x6CA3F4, "asm", "river_code", 0x802D8800],
    [0x6CA3F4, 0x7272E0, "bin"],
    [0x7272E0, 0x731E80, "asm", "volcano_code", 0x802D60E0],
    [0x731E80, 0x79F1B0, "bin"],
    [0x79F1B0, 0x7ABCD0, "asm", "valley_code", 0x802C5C20],
    [0x7ABCD0, 0x825E30, "bin"],
    [0x825E30, 0x82A280, "asm", "rainbow_code", 0x803466C0],
    [0x82A280, 0x83D730, "bin"],
    [0x83D730, 0x848BB4, "asm", "unk_end_level", 0x80369F80],
    [0x848BB4, 0x87A0B0, "bin"],
    [0x87A0B0, 0x8817E0, "asm", "unk_end_level_2", 0x801DC8C0],
    [0x8817E0, 0x8A70E0, "bin"],
    [0x8A70E0, 0x8ABEF0, "asm", "unk_end_level_3", 0x800E18C0],
    [0x8ABEF0, 0xA084B0, "bin"],
    [0xA084B0, 0xA08D10, "asm", "more_intro", 0x800E1970],
    [0xA08D10, 0xAA0650, "bin"],
    [0xAA0650, 0xAA0A54, "asm", "intro_code", 0x800E18A0],
    [0xAA0A54, 0x1000000, "bin"],
]

paper_mario_gold = [
    [0x000000, 0x000040, "header", "header"],
    [0x000040, 0x000B70, "asm",    "boot", 0xA4000040],
    [0x000B70, 0x001000, "bin",    "bootcode_font"],
    [0x1000, 0x1060, "asm", "code_1000_len_60", 0x80025c00],
    [0x1060, 0x1370, "asm", "code_1060_len_310", 0x80025c60],
    [0x1370, 0x1b40, "asm", "code_1370_len_7d0", 0x80025f70],
    [0x1b40, 0x3bf0, "asm", "code_1b40_len_20b0", 0x80026740],
    [0x3bf0, 0x42e0, "asm", "code_3bf0_len_6f0", 0x800287f0],
    [0x42e0, 0x6240, "asm", "code_42e0_len_1f60", 0x80028ee0],
    [0x6240, 0x6e40, "asm", "code_6240_len_c00", 0x8002ae40],
    [0x6e40, 0x7340, "asm", "code_6e40_len_500", 0x8002ba40],
    [0x7340, 0x7e40, "asm", "code_7340_len_b00", 0x8002bf40],
    [0x7e40, 0x8230, "asm", "code_7e40_len_3f0", 0x8002ca40],
    [0x8230, 0x9d10, "asm", "code_8230_len_1ae0", 0x8002ce30],
    [0x9d10, 0xad90, "asm", "code_9d10_len_1080", 0x8002e910],
    [0xad90, 0xd610, "asm", "code_ad90_len_2880", 0x8002f990],
    [0xd610, 0xe940, "asm", "code_d610_len_1330", 0x80032210],
    [0xe940, 0xebd0, "asm", "code_e940_len_290", 0x80033540],
    [0xebd0, 0xf270, "asm", "code_ebd0_len_6a0", 0x800337d0],
    [0xf270, 0x10400, "asm", "code_f270_len_1190", 0x80033e70],
    [0x10400, 0x11130, "asm", "code_10400_len_d30", 0x80035000],
    [0x11130, 0x111f0, "asm", "code_11130_len_c0", 0x80035d30],
    [0x111f0, 0x11a50, "asm", "code_111f0_len_860", 0x80035df0],
    [0x11a50, 0x121f0, "asm", "code_11a50_len_7a0", 0x80036650],
    [0x121f0, 0x13480, "asm", "code_121f0_len_1290", 0x80036df0],
    [0x13480, 0x13870, "asm", "code_13480_len_3f0", 0x80038080],
    [0x13870, 0x1a1f0, "asm", "code_13870_len_6980", 0x80038470],
    [0x1a1f0, 0x1f580, "asm", "code_1a1f0_len_5390", 0x8003edf0],
    [0x1f580, 0x20ec0, "asm", "code_1f580_len_1940", 0x80044180],
    [0x20ec0, 0x25f00, "asm", "code_20ec0_len_5040", 0x80045ac0],
    [0x25f00, 0x26840, "asm", "code_25f00_len_940", 0x8004ab00],
    [0x26840, 0x28910, "asm", "code_26840_len_20d0", 0x8004b440],
    [0x28910, 0x2d9a0, "asm", "code_28910_len_5090", 0x8004d510],
    [0x2d9a0, 0x2e230, "asm", "code_2d9a0_len_890", 0x800525a0],
    [0x2e230, 0x303c0, "asm", "code_2e230_len_2190", 0x80052e30],
    [0x303c0, 0x341d0, "asm", "code_303c0_len_3e10", 0x80054fc0],
    [0x341d0, 0x362a0, "asm", "code_341d0_len_20d0", 0x80058dd0],
    [0x362a0, 0x39210, "asm", "code_362a0_len_2f70", 0x8005aea0],
    [0x39210, 0x39cb0, "asm", "code_39210_len_aa0", 0x8005de10],
    [0x39cb0, 0x39db0, "asm", "code_39cb0_len_100", 0x8005e8b0],
    [0x39db0, 0x3a650, "asm", "code_39db0_len_8a0", 0x8005e9b0],
    [0x3a650, 0x3a690, "asm", "code_3a650_len_40", 0x8005f250],
    [0x3a690, 0x3a6d0, "asm", "code_3a690_len_40", 0x8005f290],
    [0x3a6d0, 0x3a6f0, "asm", "code_3a6d0_len_20", 0x8005f2d0],
    [0x3a6f0, 0x3a800, "asm", "code_3a6f0_len_110", 0x8005f2f0],
    [0x3a800, 0x3a830, "asm", "code_3a800_len_30", 0x8005f400],
    [0x3a830, 0x3a850, "asm", "code_3a830_len_20", 0x8005f430],
    [0x3a850, 0x3b290, "asm", "code_3a850_le_a40", 0x8005f450],
    [0x3b290, 0x3b390, "asm", "code_3b290_len_100", 0x8005fe90],
    [0x3b390, 0x3b4a0, "asm", "code_3b390_len_110", 0x8005ff90],
    [0x3b4a0, 0x3b710, "asm", "code_3b4a0_len_270", 0x800600a0],
    [0x3b710, 0x3b750, "asm", "code_3b710_len_40", 0x80060310],
    [0x3b750, 0x3b770, "asm", "code_3b750_len_20", 0x80060350],
    [0x3b770, 0x3b7f0, "asm", "code_3b770_len_80", 0x80060370],
    [0x3b7f0, 0x3b910, "asm", "code_3b7f0_len_120", 0x800603f0],
    [0x3b910, 0x3bd20, "asm", "code_3b910_len_410", 0x80060510],
    [0x3bd20, 0x3bde0, "asm", "code_3bd20_len_c0", 0x80060920],
    [0x3bde0, 0x3c1c0, "asm", "code_3bde0_len_3e0", 0x800609e0],
    [0x3c1c0, 0x3c220, "asm", "code_3c1c0_len_60", 0x80060dc0],
    [0x3c220, 0x3c2c0, "asm", "code_3c220_len_a0", 0x80060e20],
    [0x3c2c0, 0x3c310, "asm", "code_3c2c0_len_50", 0x80060ec0],
    [0x3c310, 0x3c490, "asm", "code_3c310_len_180", 0x80060f10],
    [0x3c490, 0x3c850, "asm", "code_3c490_len_3c0", 0x80061090],
    [0x3c850, 0x3c940, "asm", "code_3c850_len_f0", 0x80061450],
    [0x3c940, 0x3ca80, "asm", "code_3c940_len_140", 0x80061540],
    [0x3ca80, 0x3cb20, "asm", "code_3ca80_len_a0", 0x80061680],
    [0x3cb20, 0x3cc50, "asm", "code_3cb20_len_130", 0x80061720],
    [0x3cc50, 0x3ccd0, "asm", "code_3cc50_len_80", 0x80061850],
    [0x3ccd0, 0x3cd00, "asm", "code_3ccd0_len_30", 0x800618d0],
    [0x3cd00, 0x3d2f0, "asm", "code_3cd00_len_5f0", 0x80061900],
    [0x3d2f0, 0x3d300, "asm", "code_3d2f0_len_10", 0x80061ef0],
    [0x3d300, 0x3d330, "asm", "code_3d300_len_30", 0x80061f00],
    [0x3d330, 0x3dcc0, "asm", "code_3d330_len_990", 0x80061f30],
    [0x3dcc0, 0x3e720, "asm", "code_3dcc0_len_a60", 0x800628c0],
    [0x3e720, 0x3ed10, "asm", "code_3e720_len_5f0", 0x80063320],
    [0x3ed10, 0x3f310, "asm", "code_3ed10_len_600", 0x80063910],
    [0x3f310, 0x3f6d0, "asm", "code_3f310_len_3c0", 0x80063f10],
    [0x3f6d0, 0x3f9f0, "asm", "code_3f6d0_len_320", 0x800642d0],
    [0x3f9f0, 0x3fa50, "asm", "code_3f9f0_len_60", 0x800645f0],
    [0x3fa50, 0x409b0, "asm", "code_3fa50_len_f60", 0x80064650],
    [0x409b0, 0x40af0, "asm", "code_409b0_len_140", 0x800655b0],
    [0x40af0, 0x40c20, "asm", "code_40af0_len_130", 0x800656f0],
    [0x40c20, 0x40e10, "asm", "code_40c20_len_1f0", 0x80065820],
    [0x40e10, 0x41050, "asm", "code_40e10_len_240", 0x80065a10],
    [0x41050, 0x41170, "asm", "code_41050_len_120", 0x80065c50],
    [0x41170, 0x41260, "asm", "code_41170_len_f0", 0x80065d70],
    [0x41260, 0x41350, "asm", "code_41260_len_f0", 0x80065e60],
    [0x41350, 0x41420, "asm", "code_41350_len_d0", 0x80065f50],
    [0x41420, 0x41540, "asm", "code_41420_len_120", 0x80066020],
    [0x41540, 0x41600, "asm", "code_41540_len_c0", 0x80066140],
    [0x41600, 0x41640, "asm", "code_41600_len_40", 0x80066200],
    [0x41640, 0x41690, "asm", "code_41640_len_50", 0x80066240],
    [0x41690, 0x416a0, "asm", "code_41690_len_10", 0x80066290],
    [0x416a0, 0x41730, "asm", "code_416a0_len_90", 0x800662a0],
    [0x41730, 0x41750, "asm", "code_41730_len_20", 0x80066330],
    [0x41750, 0x41ba0, "asm", "code_41750_len_450", 0x80066350],
    [0x41ba0, 0x41c60, "asm", "code_41ba0_len_c0", 0x800667a0],
    [0x41c60, 0x41d20, "asm", "code_41c60_len_c0", 0x80066860],
    [0x41d20, 0x41d60, "asm", "code_41d20_len_40", 0x80066920],
    [0x41d60, 0x41db0, "asm", "code_41d60_len_50", 0x80066960],
    [0x41db0, 0x41df0, "asm", "code_41db0_len_40", 0x800669b0],
    [0x41df0, 0x41e30, "asm", "code_41df0_len_40", 0x800669f0],
    [0x41e30, 0x41e70, "asm", "code_41e30_len_40", 0x80066a30],
    [0x41e70, 0x421b0, "asm", "code_41e70_len_340", 0x80066a70],
    [0x421b0, 0x42210, "asm", "code_421b0_len_60", 0x80066db0],
    [0x42210, 0x42260, "asm", "code_42210_len_50", 0x80066e10],
    [0x42260, 0x423d0, "asm", "code_42260_len_170", 0x80066e60],
    [0x423d0, 0x42420, "asm", "code_423d0_len_50", 0x80066fd0],
    [0x42420, 0x42470, "asm", "code_42420_len_50", 0x80067020],
    [0x42470, 0x42780, "asm", "code_42470_len_310", 0x80067070],
    [0x42780, 0x428a0, "asm", "code_42780_len_120", 0x80067380],
    [0x428a0, 0x42900, "asm", "code_428a0_len_60", 0x800674a0],
    [0x42900, 0x42980, "asm", "code_42900_len_80", 0x80067500],
    [0x42980, 0x42a00, "asm", "code_42980_len_80", 0x80067580],
    [0x42a00, 0x42b00, "asm", "code_42a00_len_100", 0x80067600],
    [0x42b00, 0x42bc0, "asm", "code_42b00_len_c0", 0x80067700],
    [0x42bc0, 0x42c20, "asm", "code_42bc0_len_60", 0x800677c0],
    [0x42c20, 0x42d00, "asm", "code_42c20_len_e0", 0x80067820],
    [0x42d00, 0x42d60, "asm", "code_42d00_len_60", 0x80067900],
    [0x42d60, 0x42de0, "asm", "code_42d60_len_80", 0x80067960],
    [0x42de0, 0x42e70, "asm", "code_42de0_len_90", 0x800679e0],
    [0x42e70, 0x43200, "asm", "code_42e70_len_390", 0x80067a70],
    [0x43200, 0x439c0, "asm", "code_43200_len_7c0", 0x80067e00],
    [0x439c0, 0x43be0, "asm", "code_439c0_len_220", 0x800685c0],
    [0x43be0, 0x441c0, "asm", "code_43be0_len_5e0", 0x800687e0],
    [0x441c0, 0x44380, "asm", "code_441c0_len_1c0", 0x80068dc0],
    [0x44380, 0x44860, "asm", "code_44380_len_4e0", 0x80068f80],
    [0x44860, 0x448c0, "asm", "code_44860_len_60", 0x80069460],
    [0x448c0, 0x44ad0, "asm", "code_448c0_len_210", 0x800694c0],
    [0x44ad0, 0x44b40, "asm", "code_44ad0_len_70", 0x800696d0],
    [0x44b40, 0x455d0, "asm", "code_44b40_len_a90", 0x80069740],
    [0x455d0, 0x457c0, "asm", "code_455d0_len_1f0", 0x8006a1d0],
    [0x457c0, 0x45a30, "asm", "code_457c0_len_270", 0x8006a3c0],
    [0x45a30, 0x45a50, "asm", "code_45a30_len_20", 0x8006a630],
    [0x45a50, 0x45df0, "asm", "code_45a50_len_3a0", 0x8006a650],
    [0x45df0, 0x46760, "asm", "code_45df0_len_970", 0x8006a9f0],
    [0x46760, 0x467f0, "asm", "code_46760_len_90", 0x8006b360],
    [0x467f0, 0x46880, "asm", "code_467f0_len_90", 0x8006b3f0],
    [0x46880, 0x46ae0, "asm", "code_46880_len_260", 0x8006b480],
    [0x46ae0, 0x46c50, "asm", "code_46ae0_len_170", 0x8006b6e0],
    [0x46c50, 0x46c70, "asm", "code_46c50_len_20", 0x8006b850],
    [0x46c70, 0x47a70, "asm", "code_46c70_len_e00", 0x8006b870],
    [0x47a70, 0x47a80, "asm", "code_47a70_len_10", 0x8006c670],
    [0x47a80, 0x47a90, "asm", "code_47a80_len_10", 0x8006c680],
    [0x47a90, 0x47ae0, "asm", "code_47a90_len_50", 0x8006c690],
    [0x47ae0, 0x47bc0, "asm", "code_47ae0_len_e0", 0x8006c6e0],
    [0x47bc0, 0x47c10, "asm", "code_47bc0_len_50", 0x8006c7c0],
    [0x47c10, 0x47c60, "asm", "code_47c10_len_50", 0x8006c810],
    [0x47c60, 0x47d50, "asm", "code_47c60_len_f0", 0x8006c860],
    [0x47d50, 0x47e30, "asm", "code_47d50_len_e0", 0x8006c950],
    [0x47e30, 0x47e90, "asm", "code_47e30_len_60", 0x8006ca30],
    [0x47e90, 0x47fa0, "asm", "code_47e90_len_110", 0x8006ca90],
    [0x47fa0, 0x47fc0, "asm", "code_47fa0_len_20", 0x8006cba0],
    [0x47fc0, 0x48020, "asm", "code_47fc0_len_60", 0x8006cbc0],
    [0x48020, 0x48a20, "asm", "code_48020_len_a00", 0x8006cc20],
    [0x48a20, 0x48be0, "asm", "code_48a20_len_1c0", 0x8006d620],
    [0x48be0, 0x48c00, "asm", "code_48be0_len_20", 0x8006d7e0],
    [0x48c00, 0x491c0, "asm", "code_48c00_len_5c0", 0x8006d800],
    [0x491c0, 0x4a140, "asm", "code_491c0_len_f80", 0x8006ddc0],
    [0x4a140, 0x4a1b0, "asm", "code_4a140_len_70", 0x8006ed40],
    [0x4a1b0, 0x4a1f0, "asm", "code_4a1b0_len_40", 0x8006edb0],
    [0x4a1f0, 0x4a360, "asm", "code_4a1f0_len_170", 0x8006edf0],
    [0x4a360, 0x4ac90, "asm", "code_4a360_len_930", 0x8006ef60],
    [0x4ac90, 0x4e5a0, "asm", "code_4ac90_len_3910", 0x8006f890],
    [0x4E5A0, 0x759B0, "bin", "bin_759b0_len_27410"],
    [0x759b0, 0x90FE0, "asm", "giantchunk", 0x800DC500],
    [0x90FE0, 0xa5dd0, "bin", "giantchunk_bin", 0x800F7B30],
    [0xa5dd0, 0xb72b0, "asm", "code_a5dd0_len_114e0", 0x8010f6d0],
    [0xb72b0, 0xcd180, "asm", "code_b72b0_len_15ed0", 0x80120bb0],
    [0xcd180, 0xd0a70, "asm", "code_cd180_len_38f0", 0x80136a80],
    [0xd0a70, 0xd5a50, "asm", "code_d0a70_len_4fe0", 0x8013a370],
    [0xd5a50, 0xdba20, "asm", "code_d5a50_len_5fd0", 0x8013f350],
    [0xdba20, 0xdbd70, "asm", "code_dba20_len_350", 0x80145320],
    [0xdbd70, 0xdc470, "asm", "code_dbd70_len_700", 0x80145670],
    [0xdc470, 0xdd930, "asm", "code_dc470_len_14c0", 0x80145d70],
    [0xdd930, 0xddaf0, "asm", "code_dd930_len_1c0", 0x80147230],
    [0xddaf0, 0xde740, "asm", "code_ddaf0_len_c50", 0x801473f0],
    [0xde740, 0xe0b30, "asm", "code_de740_len_23f0", 0x80148040],
    [0xe0b30, 0xe16b0, "asm", "code_e0b30_len_b80", 0x8014a430],
    [0xe16b0, 0xe79b0, "bin", "bin_e5dd0"],
    [0xe79b0, 0xe92d0, "asm", "code_e79b0_len_1920", 0x802c3000],
    [0xe92d0, 0xef070, "asm", "code_e92d0_len_5da0", 0x802c4920],
    [0xef070, 0xf2470, "asm", "code_ef070_len_3400", 0x802ca6c0],
    [0xf2470, 0xf4c60, "asm", "code_f2470_len_27f0", 0x802cdac0],
    [0xf4c60, 0xf8f60, "asm", "code_f4c60_len_4300", 0x802d02b0],
    [0xf8f60, 0xfa4c0, "asm", "code_f8f60_len_1560", 0x802d45b0],
    [0xfa4c0, 0xfe0b0, "asm", "code_fa4c0_len_3bf0", 0x802d5b10],
    [0xfe0b0, 0xfe650, "asm", "code_fe0b0_len_5a0", 0x802d9700],
    [0xFE650, 0xFEE30, "bin", "bin_fee30_len_7e0"],
    [0xfee30, 0x101b90, "asm", "code_fee30_len_2d60", 0x802dbd40],
    [0x101b90, 0x102480, "asm", "code_101b90_len_8f0", 0x802deaa0],
    [0x102480, 0x102610, "bin", "bin_102610_len_190"],
    [0x102610, 0x104940, "asm", "code_102610_len_2330", 0x802E0D90],
    [0x104940, 0x105700, "asm", "code_104940_len_dc0", 0x802E30C0],
    [0x105700, 0x107830, "asm", "code_105700_len_2130", 0x802E3E80],
    [0x107830, 0x1086a0, "asm", "code_107830_len_e70", 0x802E5FB0],
    [0x1086a0, 0x109660, "asm", "code_1086a0_len_fc0", 0x802E6E20],
    [0x109660, 0x10A9F0, "asm", "code_109660_len_1270", 0x802E7DE0],
    [0x10A9F0, 0x163400, "bin", "bin_10A9F0"],
    [0x163400, 0x16a3e0, "asm", "code_163400", 0x80242BA0],
    [0x16a3e0, 0x16C8E0, "bin", "bin_16a3e0"],
    [0x16C8E0, 0x1AF2C0, "asm", "code_16c8e0", 0x8023E000],
    [0x1AF2C0, 0x3169f0, "bin", "bin_1AF2C0"],
    [0x3169f0, 0x316a70, "asm", "code_3169f0", 0x80200000],
    [0x316a70, 0x316c00, "asm", "code_316a70", 0x80200080],
    [0x316c00, 0x316d90, "bin", "bin_316c00"],
    [0x316d90, 0x316f30, "asm", "code_316d90", 0x802AE000],
    [0x316f30, 0x317020, "asm", "code_316f30", 0x802B2000],
    [0x317020, 0x415D90, "bin", "bin_317020"],
    [0x415D90, 0x4200C0, "asm", "code_415D90", 0x802A1000],
    [0x4200C0, 0x7e0e80, "bin", "bin_4200C0"],
    [0x7e0e80, 0x7e4d00, "asm", "code_7e0e80", 0x80280000],
    [0x7e4d00, 0xe20eb0, "bin", "bin_7e4d00"],
    [0xe20eb0, 0xe215c0, "asm", "code_e20eb0", 0x802B7000],
    [0xe215c0, 0xF007C0, "bin", "bin_19e09a8"],
    [0xF007C0, 0xF02160, "bgm", "Battle_Fanfare_02"],
    [0xF02160, 0xF03740, "bgm", "Hey_You_03"],
    [0xF03740, 0xF043F0, "bgm", "The_Goomba_King_s_Decree_07"],
    [0xF043F0, 0xF073C0, "bgm", "Attack_of_the_Koopa_Bros_08"],
    [0xF073C0, 0xF08D40, "bgm", "Trojan_Bowser_09"],
    [0xF08D40, 0xF09600, "bgm", "Chomp_Attack_0A"],
    [0xF09600, 0xF0A550, "bgm", "Ghost_Gulping_0B"],
    [0xF0A550, 0xF0BAE0, "bgm", "Keeping_Pace_0C"],
    [0xF0BAE0, 0xF0DEC0, "bgm", "Go_Mario_Go_0D"],
    [0xF0DEC0, 0xF0FD20, "bgm", "Huffin_and_Puffin_0E"],
    [0xF0FD20, 0xF110D0, "bgm", "Freeze_0F"],
    [0xF110D0, 0xF116C0, "bgm", "Winning_a_Battle_8B"],
    [0xF116C0, 0xF12320, "bgm", "Winning_a_Battle_and_Level_Up_8E"],
    [0xF12320, 0xF13C20, "bgm", "Jr_Troopa_Battle_04"],
    [0xF13C20, 0xF15F40, "bgm", "Final_Bowser_Battle_interlude_05"],
    [0xF15F40, 0xF16F80, "bgm", "Master_Battle_2C"],
    [0xF16F80, 0xF171D0, "bgm", "Game_Over_87"],
    [0xF171D0, 0xF17370, "bgm", "Resting_at_the_Toad_House_88"],
    [0xF17370, 0xF17570, "bgm", "Running_around_the_Heart_Pillar_in_Ch1_84"],
    [0xF17570, 0xF18940, "bgm", "Tutankoopa_s_Warning_45"],
    [0xF18940, 0xF193D0, "bgm", "Kammy_Koopa_s_Theme_46"],
    [0xF193D0, 0xF19BC0, "bgm", "Jr_Troopa_s_Theme_47"],
    [0xF19BC0, 0xF1A6F0, "bgm", "Goomba_King_s_Theme_50"],
    [0xF1A6F0, 0xF1ABD0, "bgm", "Koopa_Bros_Defeated_51"],
    [0xF1ABD0, 0xF1C810, "bgm", "Koopa_Bros_Theme_52"],
    [0xF1C810, 0xF1DBF0, "bgm", "Tutankoopa_s_Warning_2_53"],
    [0xF1DBF0, 0xF1F2E0, "bgm", "Tutankoopa_s_Theme_54"],
    [0xF1F2E0, 0xF20FF0, "bgm", "Tubba_Blubba_s_Theme_55"],
    [0xF20FF0, 0xF21780, "bgm", "General_Guy_s_Theme_56"],
    [0xF21780, 0xF22A00, "bgm", "Lava_Piranha_s_Theme_57"],
    [0xF22A00, 0xF23A00, "bgm", "Huff_N_Puff_s_Theme_58"],
    [0xF23A00, 0xF24810, "bgm", "Crystal_King_s_Theme_59"],
    [0xF24810, 0xF25240, "bgm", "Blooper_s_Theme_5A"],
    [0xF25240, 0xF26260, "bgm", "Midboss_Theme_5B"],
    [0xF26260, 0xF27840, "bgm", "Monstar_s_Theme_5C"],
    [0xF27840, 0xF27E20, "bgm", "Moustafa_s_Theme_86"],
    [0xF27E20, 0xF28E20, "bgm", "Fuzzy_Searching_Minigame_85"],
    [0xF28E20, 0xF29AC0, "bgm", "Phonograph_in_Mansion_44"],
    [0xF29AC0, 0xF2E130, "bgm", "Toad_Town_00"],
    [0xF2E130, 0xF2EF90, "bgm", "Bill_Blaster_Theme_48"],
    [0xF2EF90, 0xF30590, "bgm", "Monty_Mole_Theme_in_Flower_Fields_49"],
    [0xF30590, 0xF318B0, "bgm", "Shy_Guys_in_Toad_Town_4A"],
    [0xF318B0, 0xF32220, "bgm", "Whale_s_Problem_4C"],
    [0xF32220, 0xF33060, "bgm", "Toad_Town_Sewers_4B"],
    [0xF33060, 0xF33AA0, "bgm", "Unused_Theme_4D"],
    [0xF33AA0, 0xF33F10, "bgm", "Mario_s_House_Prologue_3E"],
    [0xF33F10, 0xF354E0, "bgm", "Peach_s_Party_3F"],
    [0xF354E0, 0xF35ED0, "bgm", "Goomba_Village_01"],
    [0xF35ED0, 0xF36690, "bgm", "Pleasant_Path_11"],
    [0xF36690, 0xF379E0, "bgm", "Fuzzy_s_Took_My_Shell_12"],
    [0xF379E0, 0xF38570, "bgm", "Koopa_Village_13"],
    [0xF38570, 0xF39160, "bgm", "Koopa_Bros_Fortress_14"],
    [0xF39160, 0xF3A0D0, "bgm", "Dry_Dry_Ruins_18"],
    [0xF3A0D0, 0xF3A450, "bgm", "Dry_Dry_Ruins_Mystery_19"],
    [0xF3A450, 0xF3AF20, "bgm", "Mt_Rugged_16"],
    [0xF3AF20, 0xF3C130, "bgm", "Dry_Dry_Desert_Oasis_17"],
    [0xF3C130, 0xF3CCC0, "bgm", "Dry_Dry_Outpost_15"],
    [0xF3CCC0, 0xF3E130, "bgm", "Forever_Forest_1A"],
    [0xF3E130, 0xF3F3E0, "bgm", "Boo_s_Mansion_1B"],
    [0xF3F3E0, 0xF40F00, "bgm", "Bow_s_Theme_1C"],
    [0xF40F00, 0xF42F30, "bgm", "Gusty_Gulch_Adventure_1D"],
    [0xF42F30, 0xF45500, "bgm", "Tubba_Blubba_s_Castle_1E"],
    [0xF45500, 0xF465E0, "bgm", "The_Castle_Crumbles_1F"],
    [0xF465E0, 0xF474A0, "bgm", "Shy_Guy_s_Toy_Box_20"],
    [0xF474A0, 0xF47E10, "bgm", "Toy_Train_Travel_21"],
    [0xF47E10, 0xF48410, "bgm", "Big_Lantern_Ghost_s_Theme_22"],
    [0xF48410, 0xF4A880, "bgm", "Jade_Jungle_24"],
    [0xF4A880, 0xF4BC00, "bgm", "Deep_Jungle_25"],
    [0xF4BC00, 0xF4E690, "bgm", "Lavalava_Island_26"],
    [0xF4E690, 0xF50A00, "bgm", "Search_for_the_Fearsome_5_27"],
    [0xF50A00, 0xF52520, "bgm", "Raphael_the_Raven_28"],
    [0xF52520, 0xF55C80, "bgm", "Hot_Times_in_Mt_Lavalava_29"],
    [0xF55C80, 0xF58ED0, "bgm", "Escape_from_Mt_Lavalava_2A"],
    [0xF58ED0, 0xF592B0, "bgm", "Cloudy_Climb_32"],
    [0xF592B0, 0xF5AFF0, "bgm", "Puff_Puff_Machine_33"],
    [0xF5AFF0, 0xF5C8D0, "bgm", "Flower_Fields_30"],
    [0xF5C8D0, 0xF5DF40, "bgm", "Flower_Fields_Sunny_31"],
    [0xF5DF40, 0xF5F500, "bgm", "Sun_s_Tower_34"],
    [0xF5F500, 0xF61700, "bgm", "Sun_s_Celebration_35"],
    [0xF61700, 0xF62E50, "bgm", "Shiver_City_38"],
    [0xF62E50, 0xF64220, "bgm", "Detective_Mario_39"],
    [0xF64220, 0xF64CB0, "bgm", "Snow_Road_3A"],
    [0xF64CB0, 0xF65B30, "bgm", "Over_Shiver_Mountain_3B"],
    [0xF65B30, 0xF66690, "bgm", "Starborn_Valley_3C"],
    [0xF66690, 0xF66B70, "bgm", "Sanctuary_3D"],
    [0xF66B70, 0xF67F80, "bgm", "Crystal_Palace_37"],
    [0xF67F80, 0xF69640, "bgm", "Star_Haven_60"],
    [0xF69640, 0xF6A050, "bgm", "Shooting_Star_Summit_61"],
    [0xF6A050, 0xF6C270, "bgm", "Legendary_Star_Ship_62"],
    [0xF6C270, 0xF6CED0, "bgm", "Star_Sanctuary_63"],
    [0xF6CED0, 0xF6EE40, "bgm", "Bowser_s_Castle_-_Caves_65"],
    [0xF6EE40, 0xF73390, "bgm", "Bowser_s_Castle_64"],
    [0xF73390, 0xF751F0, "bgm", "Star_Elevator_2B"],
    [0xF751F0, 0xF759C0, "bgm", "Goomba_Bros_Defeated_7E"],
    [0xF759C0, 0xF77200, "bgm", "Farewell_Twink_70"],
    [0xF77200, 0xF77680, "bgm", "Peach_Cooking_71"],
    [0xF77680, 0xF78600, "bgm", "Gourmet_Guy_72"],
    [0xF78600, 0xF79070, "bgm", "Hope_on_the_Balcony_Peach_1_73"],
    [0xF79070, 0xF7A0C0, "bgm", "Peach_s_Theme_2_74"],
    [0xF7A0C0, 0xF7AA40, "bgm", "Peach_Sneaking_75"],
    [0xF7AA40, 0xF7AD90, "bgm", "Peach_Captured_76"],
    [0xF7AD90, 0xF7BEA0, "bgm", "Quiz_Show_Intro_77"],
    [0xF7BEA0, 0xF7C780, "bgm", "Unconscious_Mario_78"],
    [0xF7C780, 0xF7DC00, "bgm", "Petunia_s_Theme_89"],
    [0xF7DC00, 0xF7E190, "bgm", "Flower_Fields_Door_appears_8A"],
    [0xF7E190, 0xF7EE20, "bgm", "Beanstalk_7B"],
    [0xF7EE20, 0xF80230, "bgm", "Lakilester_s_Theme_7D"],
    [0xF80230, 0xF81260, "bgm", "The_Sun_s_Back_7F"],
    [0xF81260, 0xF82460, "bgm", "Shiver_City_in_Crisis_79"],
    [0xF82460, 0xF82D00, "bgm", "Solved_Shiver_City_Mystery_7A"],
    [0xF82D00, 0xF83DC0, "bgm", "Merlon_s_Spell_7C"],
    [0xF83DC0, 0xF85590, "bgm", "Bowser_s_Theme_66"],
    [0xF85590, 0xF860E0, "bgm", "Train_Travel_80"],
    [0xF860E0, 0xF87000, "bgm", "Whale_Trip_81"],
    [0xF87000, 0xF87610, "bgm", "Chanterelle_s_Song_8C"],
    [0xF87610, 0xF88B30, "bgm", "Boo_s_Game_8D"],
    [0xF88B30, 0xF89570, "bgm", "Dry_Dry_Ruins_rises_up_83"],
    [0xF89570, 0xF8AAF0, "bgm", "End_of_Chapter_40"],
    [0xF8AAF0, 0xF8B820, "bgm", "Beginning_of_Chapter_41"],
    [0xF8B820, 0xF8BD90, "bgm", "Hammer_and_Jump_Upgrade_42"],
    [0xF8BD90, 0xF8C360, "bgm", "Found_Baby_Yoshi_s_4E"],
    [0xF8C360, 0xF8D110, "bgm", "New_Partner_JAP_96"],
    [0xF8D110, 0xF8D3E0, "bgm", "Unused_YI_Fanfare_4F"],
    [0xF8D3E0, 0xF90880, "bgm", "Unused_YI_Fanfare_2_5D"],
    [0xF90880, 0xF92A50, "bgm", "Peach_s_Castle_inside_Bubble_5E"],
    [0xF92A50, 0xF95510, "bgm", "Angry_Bowser_67"],
    [0xF95510, 0xF96280, "bgm", "Bowser_s_Castle_explodes_5F"],
    [0xF96280, 0xF98520, "bgm", "Peach_s_Wish_68"],
    [0xF98520, 0xF98F90, "bgm", "File_Select_69"],
    [0xF98F90, 0xF9B830, "bgm", "Title_Screen_6A"],
    [0xF9B830, 0xF9D3B0, "bgm", "Peach_s_Castle_in_Crisis_6B"],
    [0xF9D3B0, 0xF9D690, "bgm", "Mario_falls_from_Bowser_s_Castle_6C"],
    [0xF9D690, 0xF9EF30, "bgm", "Peach_s_Arrival_6D"],
    [0xF9EF30, 0xF9FA30, "bgm", "Star_Rod_Recovered_6F"],
    [0xF9FA30, 0xFA08A0, "bgm", "Mario_s_House_94"],
    [0xFA08A0, 0xFA3C60, "bgm", "Bowser_s_Attacks_95"],
    [0xFA3C60, 0xFA85F0, "bgm", "End_Parade_1_90"],
    [0xFA85F0, 0xFABE90, "bgm", "End_Parade_2_91"],
    [0xFABE90, 0xFACC80, "bgm", "The_End_6E"],
    [0xFACC80, 0xFAD210, "bgm", "Koopa_Radio_Station_2D"],
    [0xFAD210, 0xFAD8F0, "bgm", "The_End_Low_Frequency__2E"],
    [0xFAD8F0, 0xFADE70, "bgm", "SMW_Remix_2F"],
    [0xFADE70, 0xFAE860, "bgm", "New_Partner_82"],
    [0xFAE860, 0x19E09A8, "bin", "bin_fae860"],
    [0x19E09A8, 0x19E1390, "yay0", "yay_19e09a8_len_1206"],
    [0x19E1390, 0x19E1890, "yay0", "yay_19e1390_len_806"],
    [0x19E1888, 0x19E2330, "yay0", "yay_19e1888_len_2324"],
    [0x19E2330, 0x19E2DE0, "yay0", "yay_19e2330_len_cc0"],
    [0x19E2DE0, 0x19E3208, "yay0", "yay_19e2de0_len_1206"],
    [0x19E3208, 0x19E3BA8, "yay0", "yay_19e3208_len_9a6"],
    [0x19E3BA8, 0x19E3FD8, "yay0", "yay_19e3ba8_len_456"],
    [0x19E3FD8, 0x19E4828, "yay0", "yay_19e3fd8_len_4024"],
    [0x19E4828, 0x19E4BE0, "yay0", "yay_19e4828_len_3c0"],
    [0x19E4BE0, 0x19E5758, "yay0", "yay_19e4be0_len_1416"],
    [0x19E5758, 0x19E5950, "yay0", "yay_19e5758_len_802"],
    [0x19E5950, 0x19E62A0, "yay0", "yay_19e5950_len_22a0"],
    [0x19E62A0, 0x19E67B2, "yay0", "yay_19e62a0_len_512"],
    [0x19E67B2, 0x19E6B60, "bin", "bin_19e6b60"],
    [0x19E6B60, 0x19E7528, "yay0", "yay_19e6b60_len_1406"],
    [0x19E7528, 0x19E9778, "yay0", "yay_19e7528_len_8256"],
    [0x19E9778, 0x19EAF38, "yay0", "yay_19e9778_len_2800"],
    [0x19EAF38, 0x19EC4E0, "yay0", "yay_19eaf38_len_40c0"],
    [0x19EC4E0, 0x19EDD30, "yay0", "yay_19ec4e0_len_1910"],
    [0x19EDD30, 0x19EEB18, "yay0", "yay_19edd30_len_2204"],
    [0x19EEB18, 0x19F0070, "yay0", "yay_19eeb18_len_10062"],
    [0x19F0070, 0x19F15A0, "yay0", "yay_19f0070_len_158c"],
    [0x19F15A0, 0x19F26D8, "yay0", "yay_19f15a0_len_2252"],
    [0x19F26D8, 0x19F5390, "yay0", "yay_19f26d8_len_5102"],
    [0x19F5390, 0x19F7398, "yay0", "yay_19f5390_len_2002"],
    [0x19F7398, 0x19FA128, "yay0", "yay_19f7398_len_8024"],
    [0x19FA128, 0x19FCAE8, "yay0", "yay_19fa128_len_4ac6"],
    [0x19FCAE8, 0x19FED70, "yay0", "yay_19fcae8_len_2502"],
    [0x19FED70, 0x1A00958, "yay0", "yay_19fed70_len_200004"],
    [0x1A00958, 0x1A02D00, "yay0", "yay_1a00958_len_24a2"],
    [0x1A02D00, 0x1A04400, "yay0", "yay_1a02d00_len_4000"],
    [0x1A04400, 0x1A05550, "yay0", "yay_1a04400_len_114c"],
    [0x1A05550, 0x1A06390, "yay0", "yay_1a05550_len_2280"],
    [0x1A06390, 0x1A06F98, "yay0", "yay_1a06390_len_c04"],
    [0x1A06F98, 0x1A07B68, "yay0", "yay_1a06f98_len_1066"],
    [0x1A07B68, 0x1A0A0A0, "yay0", "yay_1a07b68_len_8092"],
    [0x1A0A0A0, 0x1A0ACC8, "yay0", "yay_1a0a0a0_len_c46"],
    [0x1A0ACC8, 0x1A0B780, "yay0", "yay_1a0acc8_len_1300"],
    [0x1A0B780, 0x1A0BBE0, "yay0", "yay_1a0b780_len_85c"],
    [0x1A0BBE0, 0x1A0C000, "yay0", "yay_1a0bbe0_len_4000"],
    [0x1A0C000, 0x1A0C438, "yay0", "yay_1a0c000_len_438"],
    [0x1A0C438, 0x1A0C860, "yay0", "yay_1a0c438_len_842"],
    [0x1A0C860, 0x1A0D1E8, "yay0", "yay_1a0c860_len_1186"],
    [0x1A0D1E8, 0x1A0D5B0, "yay0", "yay_1a0d1e8_len_406"],
    [0x1A0D5B0, 0x1A0D970, "yay0", "yay_1a0d5b0_len_840"],
    [0x1A0D970, 0x1A0EF00, "yay0", "yay_1a0d970_len_268e"],
    [0x1A0EF00, 0x1A13920, "yay0", "yay_1a0ef00_len_11020"],
    [0x1A13920, 0x1A15850, "yay0", "yay_1a13920_len_404c"],
    [0x1A15850, 0x1A183F8, "yay0", "yay_1a15850_len_83a4"],
    [0x1A183F8, 0x1A1A608, "yay0", "yay_1a183f8_len_2404"],
    [0x1A1A608, 0x1A1C5E8, "yay0", "yay_1a1a608_len_41e2"],
    [0x1A1C5E8, 0x1A1D6D0, "yay0", "yay_1a1c5e8_len_1202"],
    [0x1A1D6D0, 0x1A1E478, "yay0", "yay_1a1d6d0_len_2028"],
    [0x1A1E478, 0x1A1F370, "yay0", "yay_1a1e478_len_1306"],
    [0x1A1F370, 0x1A226B0, "yay0", "yay_1a1f370_len_2048c"],
    [0x1A226B0, 0x1A249B8, "yay0", "yay_1a226b0_len_4908"],
    [0x1A249B8, 0x1A25E78, "yay0", "yay_1a249b8_len_1644"],
    [0x1A25E78, 0x1A27FF0, "yay0", "yay_1a25e78_len_2186"],
    [0x1A27FF0, 0x1A28BE0, "yay0", "yay_1a27ff0_len_800a"],
    [0x1A28BE0, 0x1A29680, "yay0", "yay_1a28be0_len_1400"],
    [0x1A29680, 0x1A2BC68, "yay0", "yay_1a29680_len_2862"],
    [0x1A2BC68, 0x1A2E120, "yay0", "yay_1a2bc68_len_4112"],
    [0x1A2E120, 0x1A2F3F8, "yay0", "yay_1a2e120_len_12d8"],
    [0x1A2F3F8, 0x1A31D18, "yay0", "yay_1a2f3f8_len_10c06"],
    [0x1A31D18, 0x1A33AB0, "yay0", "yay_1a31d18_len_22a6"],
    [0x1A33AB0, 0x1A35BB8, "yay0", "yay_1a33ab0_len_4106"],
    [0x1A35BB8, 0x1A369A8, "yay0", "yay_1a35bb8_len_2006"],
    [0x1A369A8, 0x1A39600, "yay0", "yay_1a369a8_len_9600"],
    [0x1A39600, 0x1A3B018, "yay0", "yay_1a39600_len_2014"],
    [0x1A3B018, 0x1A3C310, "yay0", "yay_1a3b018_len_4300"],
    [0x1A3C310, 0x1A3FCC8, "yay0", "yay_1a3c310_len_3cc2"],
    [0x1A3FCC8, 0x1A423D8, "yay0", "yay_1a3fcc8_len_40314"],
    [0x1A423D8, 0x1A449C0, "yay0", "yay_1a423d8_len_4822"],
    [0x1A449C0, 0x1A46568, "yay0", "yay_1a449c0_len_2422"],
    [0x1A46568, 0x1A49340, "yay0", "yay_1a46568_len_9212"],
    [0x1A49340, 0x1A4AC88, "yay0", "yay_1a49340_len_2c88"],
    [0x1A4AC88, 0x1A4D7E8, "yay0", "yay_1a4ac88_len_5362"],
    [0x1A4D7E8, 0x1A4E028, "yay0", "yay_1a4d7e8_len_2006"],
    [0x1A4E028, 0x1A4FA60, "yay0", "yay_1a4e028_len_1a56"],
    [0x1A4FA60, 0x1A531D0, "yay0", "yay_1a4fa60_len_10190"],
    [0x1A531D0, 0x1A53D48, "yay0", "yay_1a531d0_len_c08"],
    [0x1A53D48, 0x1A56C80, "yay0", "yay_1a53d48_len_4032"],
    [0x1A56C80, 0x1A58F58, "yay0", "yay_1a56c80_len_8358"],
    [0x1A58F58, 0x1A5A5A8, "yay0", "yay_1a58f58_len_20a4"],
    [0x1A5A5A8, 0x1A62B40, "yay0", "yay_1a5a5a8_len_20a14"],
    [0x1A62B40, 0x1A641F8, "yay0", "yay_1a62b40_len_40b6"],
    [0x1A641F8, 0x1A666F0, "yay0", "yay_1a641f8_len_2602"],
    [0x1A666F0, 0x1A68870, "yay0", "yay_1a666f0_len_8800"],
    [0x1A68870, 0x1A6C630, "yay0", "yay_1a68870_len_460a"],
    [0x1A6C630, 0x1A6D5A0, "yay0", "yay_1a6c630_len_118c"],
    [0x1A6D5A0, 0x1A6EF50, "yay0", "yay_1a6d5a0_len_2a4c"],
    [0x1A6EF50, 0x1A70FF0, "yay0", "yay_1a6ef50_len_100ae"],
    [0x1A70FF0, 0x1A74FC0, "yay0", "yay_1a70ff0_len_4000"],
    [0x1A74FC0, 0x1A78A40, "yay0", "yay_1a74fc0_len_803a"],
    [0x1A78A40, 0x1A79900, "yay0", "yay_1a78a40_len_1100"],
    [0x1A79900, 0x1A7D798, "yay0", "yay_1a79900_len_4698"],
    [0x1A7D798, 0x1A7EEA0, "yay0", "yay_1a7d798_len_2804"],
    [0x1A7EEA0, 0x1A7EFD8, "yay0", "yay_1a7eea0_len_158"],
    [0x1A7EFD8, 0x1A83450, "yay0", "yay_1a7efd8_len_81002"],
    [0x1A83450, 0x1A85668, "yay0", "yay_1a83450_len_4226"],
    [0x1A85668, 0x1A87958, "yay0", "yay_1a85668_len_2910"],
    [0x1A87958, 0x1A8BF98, "yay0", "yay_1a87958_len_8680"],
    [0x1A8BF98, 0x1A8FE28, "yay0", "yay_1a8bf98_len_4020"],
    [0x1A8FE28, 0x1A93EA0, "yay0", "yay_1a8fe28_len_10096"],
    [0x1A93EA0, 0x1A94188, "yay0", "yay_1a93ea0_len_4102"],
    [0x1A94188, 0x1A94480, "yay0", "yay_1a94188_len_476"],
    [0x1A94480, 0x1A946A8, "yay0", "yay_1a94480_len_226"],
    [0x1A946A8, 0x1A94A00, "yay0", "yay_1a946a8_len_956"],
    [0x1A94A00, 0x1A94C58, "yay0", "yay_1a94a00_len_456"],
    [0x1A94C58, 0x1A98D98, "yay0", "yay_1a94c58_len_8184"],
    [0x1A98D98, 0x1A9BA80, "yay0", "yay_1a98d98_len_3264"],
    [0x1A9BA80, 0x1A9DB68, "yay0", "yay_1a9ba80_len_4168"],
    [0x1A9DB68, 0x1AA0048, "yay0", "yay_1a9db68_len_20004"],
    [0x1AA0048, 0x1AA0698, "yay0", "yay_1aa0048_len_692"],
    [0x1AA0698, 0x1AA1008, "yay0", "yay_1aa0698_len_1000"],
    [0x1AA1008, 0x1AA6D58, "yay0", "yay_1aa1008_len_6d54"],
    [0x1AA6D58, 0x1AAD600, "yay0", "yay_1aa6d58_len_90a4"],
    [0x1AAD600, 0x1AB1BF0, "yay0", "yay_1aad600_len_109f0"],
    [0x1AB1BF0, 0x1AB2368, "yay0", "yay_1ab1bf0_len_2004"],
    [0x1AB2368, 0x1ABA290, "yay0", "yay_1ab2368_len_8086"],
    [0x1ABA290, 0x1AC14A8, "yay0", "yay_1aba290_len_41422"],
    [0x1AC14A8, 0x1AC3910, "yay0", "yay_1ac14a8_len_2902"],
    [0x1AC3910, 0x1ACBC98, "yay0", "yay_1ac3910_len_8488"],
    [0x1ACBC98, 0x1ACE058, "yay0", "yay_1acbc98_len_4042"],
    [0x1ACE058, 0x1ACF910, "yay0", "yay_1ace058_len_1902"],
    [0x1ACF910, 0x1AD06D8, "yay0", "yay_1acf910_len_106c8"],
    [0x1AD06D8, 0x1AD0B98, "yay0", "yay_1ad06d8_len_904"],
    [0x1AD0B98, 0x1AD1E90, "yay0", "yay_1ad0b98_len_1400"],
    [0x1AD1E90, 0x1AD2348, "yay0", "yay_1ad1e90_len_2146"],
    [0x1AD2348, 0x1AD27F8, "yay0", "yay_1ad2348_len_4b2"],
    [0x1AD27F8, 0x1AD28A8, "yay0", "yay_1ad27f8_len_800"],
    [0x1AD28A8, 0x1AD44F0, "yay0", "yay_1ad28a8_len_4446"],
    [0x1AD44F0, 0x1AD4758, "yay0", "yay_1ad44f0_len_304"],
    [0x1AD4758, 0x1AD57C0, "yay0", "yay_1ad4758_len_1080"],
    [0x1AD57C0, 0x1AD9D50, "yay0", "yay_1ad57c0_len_880c"],
    [0x1AD9D50, 0x1ADA498, "yay0", "yay_1ad9d50_len_2082"],
    [0x1ADA498, 0x1ADCFC0, "yay0", "yay_1ada498_len_4b26"],
    [0x1ADCFC0, 0x1AE2168, "yay0", "yay_1adcfc0_len_22024"],
    [0x1AE2168, 0x1AE2EA0, "yay0", "yay_1ae2168_len_e96"],
    [0x1AE2EA0, 0x1AE6A58, "yay0", "yay_1ae2ea0_len_4058"],
    [0x1AE6A58, 0x1AEB778, "yay0", "yay_1ae6a58_len_9524"],
    [0x1AEB778, 0x1AF4958, "yay0", "yay_1aeb778_len_14802"],
    [0x1AF4958, 0x1AFCB18, "yay0", "yay_1af4958_len_8202"],
    [0x1AFCB18, 0x1AFF748, "yay0", "yay_1afcb18_len_3442"],
    [0x1AFF748, 0x1B00640, "yay0", "yay_1aff748_len_100000"],
    [0x1B00640, 0x1B01390, "yay0", "yay_1b00640_len_118a"],
    [0x1B01390, 0x1B01C08, "yay0", "yay_1b01390_len_c04"],
    [0x1B01C08, 0x1B02128, "yay0", "yay_1b01c08_len_2120"],
    [0x1B02128, 0x1B02970, "yay0", "yay_1b02128_len_844"],
    [0x1B02970, 0x1B03118, "yay0", "yay_1b02970_len_1004"],
    [0x1B03118, 0x1B03C18, "yay0", "yay_1b03118_len_c04"],
    [0x1B03C18, 0x1B045E8, "yay0", "yay_1b03c18_len_41e2"],
    [0x1B045E8, 0x1B04FC0, "yay0", "yay_1b045e8_len_a12"],
    [0x1B04FC0, 0x1B05998, "yay0", "yay_1b04fc0_len_1012"],
    [0x1B05998, 0x1B06C88, "yay0", "yay_1b05998_len_2400"],
    [0x1B06C88, 0x1B07C48, "yay0", "yay_1b06c88_len_1046"],
    [0x1B07C48, 0x1B09440, "yay0", "yay_1b07c48_len_8034"],
    [0x1B09440, 0x1B0B290, "yay0", "yay_1b09440_len_228e"],
    [0x1B0B290, 0x1B0B9A0, "yay0", "yay_1b0b290_len_90c"],
    [0x1B0B9A0, 0x1B0C548, "yay0", "yay_1b0b9a0_len_4448"],
    [0x1B0C548, 0x1B0CAC0, "yay0", "yay_1b0c548_len_ab4"],
    [0x1B0CAC0, 0x1B0D130, "yay0", "yay_1b0cac0_len_1130"],
    [0x1B0D130, 0x1B0EB80, "yay0", "yay_1b0d130_len_2a4c"],
    [0x1B0EB80, 0x1B10CC0, "yay0", "yay_1b0eb80_len_1043e"],
    [0x1B10CC0, 0x1B129A0, "yay0", "yay_1b10cc0_len_2120"],
    [0x1B129A0, 0x1B13548, "yay0", "yay_1b129a0_len_1444"],
    [0x1B13548, 0x1B16420, "yay0", "yay_1b13548_len_4016"],
    [0x1B16420, 0x1B17128, "yay0", "yay_1b16420_len_1104"],
    [0x1B17128, 0x1B17840, "yay0", "yay_1b17128_len_814"],
    [0x1B17840, 0x1B19318, "yay0", "yay_1b17840_len_8316"],
    [0x1B19318, 0x1B1A030, "yay0", "yay_1b19318_len_2020"],
    [0x1B1A030, 0x1B1B8C8, "yay0", "yay_1b1a030_len_18c6"],
    [0x1B1B8C8, 0x1B1BC88, "yay0", "yay_1b1b8c8_len_402"],
    [0x1B1BC88, 0x1B1C7A0, "yay0", "yay_1b1bc88_len_4316"],
    [0x1B1C7A0, 0x1B1CD28, "yay0", "yay_1b1c7a0_len_808"],
    [0x1B1CD28, 0x1B21C48, "yay0", "yay_1b1cd28_len_21040"],
    [0x1B21C48, 0x1B23290, "yay0", "yay_1b21c48_len_2290"],
    [0x1B23290, 0x1B253E0, "yay0", "yay_1b23290_len_414c"],
    [0x1B253E0, 0x1B26660, "yay0", "yay_1b253e0_len_241c"],
    [0x1B26660, 0x1B283F8, "yay0", "yay_1b26660_len_8192"],
    [0x1B283F8, 0x1B29C60, "yay0", "yay_1b283f8_len_1c02"],
    [0x1B29C60, 0x1B2A688, "yay0", "yay_1b29c60_len_2284"],
    [0x1B2A688, 0x1B2B3E8, "yay0", "yay_1b2a688_len_1166"],
    [0x1B2B3E8, 0x1B2C8D8, "yay0", "yay_1b2b3e8_len_4810"],
    [0x1B2C8D8, 0x1B2D7B0, "yay0", "yay_1b2c8d8_len_1722"],
    [0x1B2D7B0, 0x1B2E328, "yay0", "yay_1b2d7b0_len_2006"],
    [0x1B2E328, 0x1B2ED60, "yay0", "yay_1b2e328_len_c52"],
    [0x1B2ED60, 0x1B2FA18, "yay0", "yay_1b2ed60_len_1218"],
    [0x1B2FA18, 0x1B31A18, "yay0", "yay_1b2fa18_len_10002"],
    [0x1B31A18, 0x1B33000, "yay0", "yay_1b31a18_len_25e2"],
    [0x1B33000, 0x1B34098, "yay0", "yay_1b33000_len_4096"],
    [0x1B34098, 0x1B34928, "yay0", "yay_1b34098_len_920"],
    [0x1B34928, 0x1B34C00, "yay0", "yay_1b34928_len_2d4"],
    [0x1B34C00, 0x1B35480, "yay0", "yay_1b34c00_len_107e"],
    [0x1B35480, 0x1B36440, "yay0", "yay_1b35480_len_2040"],
    [0x1B36440, 0x1B38748, "yay0", "yay_1b36440_len_8306"],
    [0x1B38748, 0x1B39A98, "yay0", "yay_1b38748_len_1894"],
    [0x1B39A98, 0x1B3A2E8, "yay0", "yay_1b39a98_len_2062"],
    [0x1B3A2E8, 0x1B3A818, "yay0", "yay_1b3a2e8_len_816"],
    [0x1B3A818, 0x1B3C488, "yay0", "yay_1b3a818_len_4482"],
    [0x1B3C488, 0x1B3CAC8, "yay0", "yay_1b3c488_len_a44"],
    [0x1B3CAC8, 0x1B3D0A0, "yay0", "yay_1b3cac8_len_1012"],
    [0x1B3D0A0, 0x1B3D920, "yay0", "yay_1b3d0a0_len_91a"],
    [0x1B3D920, 0x1B3F060, "yay0", "yay_1b3d920_len_205c"],
    [0x1B3F060, 0x1B40048, "yay0", "yay_1b3f060_len_40008"],
    [0x1B40048, 0x1B40720, "yay0", "yay_1b40048_len_714"],
    [0x1B40720, 0x1B49570, "yay0", "yay_1b40720_len_904e"],
    [0x1B49570, 0x1B4C3E8, "yay0", "yay_1b49570_len_4288"],
    [0x1B4C3E8, 0x1B4DEA0, "yay0", "yay_1b4c3e8_len_1c12"],
    [0x1B4DEA0, 0x1B4FD98, "yay0", "yay_1b4dea0_len_2116"],
    [0x1B4FD98, 0x1B50CD8, "yay0", "yay_1b4fd98_len_10040"],
    [0x1B50CD8, 0x1B51B08, "yay0", "yay_1b50cd8_len_1302"],
    [0x1B51B08, 0x1B54258, "yay0", "yay_1b51b08_len_4052"],
    [0x1B54258, 0x1B580A0, "yay0", "yay_1b54258_len_8082"],
    [0x1B580A0, 0x1B5A248, "yay0", "yay_1b580a0_len_2242"],
    [0x1B5A248, 0x1B5BB88, "yay0", "yay_1b5a248_len_1980"],
    [0x1B5BB88, 0x1B5CC90, "yay0", "yay_1b5bb88_len_4406"],
    [0x1B5CC90, 0x1B5E968, "yay0", "yay_1b5cc90_len_2168"],
    [0x1B5E968, 0x1B5ED88, "yay0", "yay_1b5e968_len_482"],
    [0x1B5ED88, 0x1B608C0, "yay0", "yay_1b5ed88_len_20040"],
    [0x1B608C0, 0x1B625F8, "yay0", "yay_1b608c0_len_2534"],
    [0x1B625F8, 0x1B633D0, "yay0", "yay_1b625f8_len_1202"],
    [0x1B633D0, 0x1B64878, "yay0", "yay_1b633d0_len_4824"],
    [0x1B64878, 0x1B657E0, "yay0", "yay_1b64878_len_1786"],
    [0x1B657E0, 0x1B65A08, "yay0", "yay_1b657e0_len_808"],
    [0x1B65A08, 0x1B65E50, "yay0", "yay_1b65a08_len_444"],
    [0x1B65E50, 0x1B66238, "yay0", "yay_1b65e50_len_2024"],
    [0x1B66238, 0x1B69580, "yay0", "yay_1b66238_len_9542"],
    [0x1B69580, 0x1B6C318, "yay0", "yay_1b69580_len_4218"],
    [0x1B6C318, 0x1B6DD98, "yay0", "yay_1b6c318_len_1c80"],
    [0x1B6DD98, 0x1B6F150, "yay0", "yay_1b6dd98_len_2046"],
    [0x1B6F150, 0x1B71618, "yay0", "yay_1b6f150_len_10608"],
    [0x1B71618, 0x1B72890, "yay0", "yay_1b71618_len_2882"],
    [0x1B72890, 0x1B73B08, "yay0", "yay_1b72890_len_1302"],
    [0x1B73B08, 0x1B747B8, "yay0", "yay_1b73b08_len_44b0"],
    [0x1B747B8, 0x1B76E30, "yay0", "yay_1b747b8_len_2800"],
    [0x1B76E30, 0x1B78EC0, "yay0", "yay_1b76e30_len_808a"],
    [0x1B78EC0, 0x1B79A20, "yay0", "yay_1b78ec0_len_101c"],
    [0x1B79A20, 0x1B79F08, "yay0", "yay_1b79a20_len_508"],
    [0x1B79F08, 0x1B7AA08, "yay0", "yay_1b79f08_len_2004"],
    [0x1B7AA08, 0x1B7B008, "yay0", "yay_1b7aa08_len_1000"],
    [0x1B7B008, 0x1B7BB50, "yay0", "yay_1b7b008_len_b44"],
    [0x1B7BB50, 0x1B7EC68, "yay0", "yay_1b7bb50_len_4424"],
    [0x1B7EC68, 0x1B7FF48, "yay0", "yay_1b7ec68_len_1300"],
    [0x1B7FF48, 0x1B81E88, "yay0", "yay_1b7ff48_len_80086"],
    [0x1B81E88, 0x1B82058, "yay0", "yay_1b81e88_len_2050"],
    [0x1B82058, 0x1B82202, "yay0", "yay_1b82058_len_202"],
    [0x1B82202, 0x1E40000, "bin", "bin_1b82202"],
    [0x1E40000, 0x27FEE22, "Map_Assets.FS", "map_assets_fs_1e40000_len_9b7cea"],
    [0x27FEE22, 0x2800000, "bin", "bin_27fee22"],
]
