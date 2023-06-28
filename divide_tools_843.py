


def divide_cdhit(dataset, fold):
    d1 = [160, 354, 392, 464, 513, 674, 537, 367, 368, 369, 652, 604, 606, 421, 422, 351, 352, 303, 112, 212, 213, 221,
          305,
          485, 486, 60, 87, 216, 249, 294, 375, 376, 306, 496, 548, 676, 465, 347, 547, 666, 477, 560, 171, 172, 721,
          516,
          611, 612, 615, 616, 617, 618, 619, 620, 621, 564, 263, 339, 340, 478, 50, 51, 252, 333, 479, 447, 266, 267,
          268,
          270, 644, 126, 170, 265, 341, 667, 673, 686, 14, 47, 80, 184, 185, 186, 326, 327, 380, 381, 430, 431, 432,
          439,
          489, 532, 533, 538, 570, 499, 197, 627, 183, 298, 304, 493, 494, 495, 299, 469, 529, 653, 77, 78, 136, 138,
          142,
          143, 484, 264, 272, 273, 530, 658, 675, 246, 247, 364, 365, 33, 39, 58, 59, 147, 295, 206, 116, 258, 262, 601,
          535,
          536, 122, 123, 562, 398, 317]
    d2 = [177, 181, 386, 491, 594, 595, 596, 597, 0, 63, 64, 65, 68, 69, 661, 348, 388, 389, 390, 656, 657, 355, 356,
          378,
          379, 573, 28, 29, 30, 156, 157, 158, 159, 255, 259, 260, 274, 288, 328, 329, 345, 408, 442, 458, 459, 466,
          467,
          468, 470, 471, 472, 473, 474, 492, 502, 503, 504, 505, 506, 507, 508, 549, 566, 567, 568, 574, 575, 579, 581,
          582,
          583, 584, 585, 608, 613, 614, 624, 625, 629, 631, 632, 633, 634, 655, 671, 710, 712, 713, 714, 717, 308, 331,
          385,
          336, 427, 428, 511, 512, 284, 593, 155, 287, 244, 289, 382, 414, 440, 451, 510, 517, 556, 635, 636, 269, 271,
          242,
          425, 445, 446, 454, 522, 609, 610, 622, 654, 693, 694, 695, 151, 168, 169, 429, 569, 623, 359, 332, 626, 628,
          630,
          716, 362, 363, 257, 711]
    d3 = [589, 590, 279, 320, 462, 349, 99, 223, 137, 220, 476, 563, 534, 664, 665, 161, 515, 526, 487, 498, 646, 660,
          699,
          22, 4, 82, 231, 253, 461, 600, 19, 76, 86, 391, 709, 475, 360, 718, 218, 175, 176, 192, 193, 275, 318, 393,
          463,
          684, 685, 542, 543, 203, 204, 205, 314, 239, 199, 240, 201, 292, 371, 396, 397, 370, 576, 577, 578, 417, 707,
          531,
          74, 75, 120, 321, 322, 553, 554, 178, 179, 280, 409, 153, 419, 420, 603, 648, 649, 650, 692, 2, 12, 366, 248,
          602,
          3, 41, 95, 519, 198, 290, 291, 443, 444, 20, 71, 480, 481, 129, 141, 217, 18, 121, 311, 312, 55, 125, 457,
          704,
          705, 15, 52, 224, 301, 558, 357, 13, 697, 233, 234, 256, 677, 243, 245, 383, 426, 453, 557, 544, 330, 374,
          219,
          113, 148, 592]
    d4 = [394, 539, 651, 523, 599, 88, 107, 182, 346, 690, 691, 403, 404, 353, 637, 645, 696, 21, 180, 412, 189, 518,
          520,
          541, 528, 173, 293, 372, 433, 434, 435, 662, 659, 227, 228, 229, 343, 680, 208, 358, 415, 319, 551, 552, 668,
          672,
          96, 97, 235, 580, 702, 706, 300, 703, 46, 83, 124, 154, 230, 236, 237, 302, 309, 698, 436, 437, 438, 70, 395,
          524,
          525, 53, 211, 670, 225, 663, 683, 605, 607, 456, 1, 6, 8, 10, 11, 16, 17, 23, 24, 25, 26, 27, 34, 35, 37, 38,
          40,
          43, 44, 45, 48, 49, 54, 56, 61, 62, 66, 67, 79, 84, 85, 93, 94, 98, 102, 103, 104, 105, 111, 115, 117, 118,
          119,
          130, 131, 132, 133, 134, 135, 202, 207, 209, 210, 226, 254, 550, 586, 587, 588, 591, 261, 296, 31, 720]
    d5 = [7, 162, 163, 164, 165, 166, 167, 313, 561, 108, 565, 678, 679, 214, 546, 286, 72, 669, 681, 682, 310, 232,
          488,
          490, 323, 324, 325, 334, 350, 455, 89, 90, 191, 399, 441, 521, 640, 641, 642, 643, 701, 276, 277, 278, 335,
          36,
          342, 344, 406, 639, 128, 241, 282, 283, 460, 555, 410, 411, 114, 145, 377, 527, 540, 708, 196, 222, 361, 719,
          100,
          638, 416, 337, 338, 144, 5, 9, 91, 92, 139, 140, 174, 215, 187, 387, 423, 424, 73, 413, 497, 42, 188, 195,
          251,
          448, 559, 687, 688, 373, 81, 190, 400, 401, 402, 501, 509, 514, 152, 297, 307, 715, 316, 127, 405, 32, 101,
          109,
          110, 450, 452, 106, 482, 483, 545, 200, 598, 571, 572, 285, 384, 194, 689, 281, 407, 418, 500, 315, 449, 57,
          700,
          647, 149, 150, 146, 238]
    folds_d = [d1, d2, d3, d4, d5]
    train_folds = [0, 1, 2, 4, 3]
    train_folds.remove(fold)
    train = []
    val = []
    for i in train_folds:
        print(folds_d[i])
        for j in folds_d[i]:
            train.append(dataset[j])
        # train.extend(folds_d[i])
    for i in folds_d[fold]:
        val.append(dataset[i])
    return train, val
