/*
 Navicat Premium Data Transfer

 Source Server         : mysql
 Source Server Type    : MySQL
 Source Server Version : 50732 (5.7.32-log)
 Source Host           : localhost:3306
 Source Schema         : connect5web

 Target Server Type    : MySQL
 Target Server Version : 50732 (5.7.32-log)
 File Encoding         : 65001

 Date: 16/06/2024 11:00:53
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for game
-- ----------------------------
DROP TABLE IF EXISTS `game`;
CREATE TABLE `game`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `player1_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `player2_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `winner_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `created_at` datetime NOT NULL,
  `ended_at` datetime NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `created_at`(`created_at`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 7 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of game
-- ----------------------------
INSERT INTO `game` VALUES (4, 'admin', 'coffeecat', 'coffeecat', '2024-06-06 20:11:46', '2024-06-06 20:12:55');
INSERT INTO `game` VALUES (5, 'admin', 'coffeecat', 'coffeecat', '2024-06-15 15:19:28', '2024-06-15 15:20:40');
INSERT INTO `game` VALUES (6, 'coffeecat', 'admin', 'admin', '2024-06-16 08:47:15', '2024-06-16 08:48:28');

-- ----------------------------
-- Table structure for move
-- ----------------------------
DROP TABLE IF EXISTS `move`;
CREATE TABLE `move`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `game_id` int(11) NOT NULL,
  `player_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `move` int(11) NOT NULL,
  `move_number` int(11) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `move_ibfk_1`(`game_id`) USING BTREE,
  INDEX `move_ibfk_2`(`player_name`) USING BTREE,
  CONSTRAINT `move_ibfk_1` FOREIGN KEY (`game_id`) REFERENCES `game` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `move_ibfk_2` FOREIGN KEY (`player_name`) REFERENCES `user` (`username`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 112 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of move
-- ----------------------------
INSERT INTO `move` VALUES (62, 4, 'admin', 128, 1);
INSERT INTO `move` VALUES (63, 4, 'coffeecat', 113, 2);
INSERT INTO `move` VALUES (64, 4, 'admin', 157, 3);
INSERT INTO `move` VALUES (65, 4, 'coffeecat', 140, 4);
INSERT INTO `move` VALUES (66, 4, 'admin', 143, 5);
INSERT INTO `move` VALUES (67, 4, 'coffeecat', 210, 6);
INSERT INTO `move` VALUES (68, 4, 'admin', 158, 7);
INSERT INTO `move` VALUES (69, 4, 'coffeecat', 155, 8);
INSERT INTO `move` VALUES (70, 4, 'admin', 173, 9);
INSERT INTO `move` VALUES (71, 4, 'coffeecat', 188, 10);
INSERT INTO `move` VALUES (72, 4, 'admin', 142, 11);
INSERT INTO `move` VALUES (73, 4, 'coffeecat', 170, 12);
INSERT INTO `move` VALUES (74, 4, 'admin', 171, 13);
INSERT INTO `move` VALUES (75, 4, 'coffeecat', 185, 14);
INSERT INTO `move` VALUES (76, 4, 'admin', 200, 15);
INSERT INTO `move` VALUES (77, 4, 'coffeecat', 125, 16);
INSERT INTO `move` VALUES (88, 4, 'admin', 113, 0);
INSERT INTO `move` VALUES (89, 4, 'admin', 157, 0);
INSERT INTO `move` VALUES (91, 5, 'admin', 113, 1);
INSERT INTO `move` VALUES (92, 5, 'coffeecat', 128, 2);
INSERT INTO `move` VALUES (93, 5, 'admin', 127, 3);
INSERT INTO `move` VALUES (94, 5, 'coffeecat', 142, 4);
INSERT INTO `move` VALUES (95, 5, 'admin', 141, 5);
INSERT INTO `move` VALUES (96, 5, 'coffeecat', 156, 6);
INSERT INTO `move` VALUES (97, 5, 'admin', 155, 7);
INSERT INTO `move` VALUES (98, 5, 'coffeecat', 170, 8);
INSERT INTO `move` VALUES (99, 5, 'admin', 169, 9);
INSERT INTO `move` VALUES (100, 5, 'admin', 113, 0);
INSERT INTO `move` VALUES (101, 5, 'admin', 128, 0);
INSERT INTO `move` VALUES (102, 6, 'coffeecat', 113, 1);
INSERT INTO `move` VALUES (103, 6, 'admin', 128, 2);
INSERT INTO `move` VALUES (104, 6, 'coffeecat', 143, 3);
INSERT INTO `move` VALUES (105, 6, 'admin', 129, 4);
INSERT INTO `move` VALUES (106, 6, 'coffeecat', 112, 5);
INSERT INTO `move` VALUES (107, 6, 'admin', 127, 6);
INSERT INTO `move` VALUES (108, 6, 'coffeecat', 111, 7);
INSERT INTO `move` VALUES (109, 6, 'admin', 126, 8);
INSERT INTO `move` VALUES (110, 6, 'coffeecat', 110, 9);
INSERT INTO `move` VALUES (111, 6, 'admin', 125, 10);

-- ----------------------------
-- Table structure for train_data
-- ----------------------------
DROP TABLE IF EXISTS `train_data`;
CREATE TABLE `train_data`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `used` text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `board` json NOT NULL,
  `move` int(11) NOT NULL,
  `win` text CHARACTER SET utf8mb4 COLLATE utf8mb4_danish_ci NOT NULL,
  `move_number` int(11) NOT NULL,
  `game_id` int(11) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `game_id`(`game_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 65 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of train_data
-- ----------------------------
INSERT INTO `train_data` VALUES (1, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}]', 141, 'true', 5, 3);
INSERT INTO `train_data` VALUES (2, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}]', 111, 'false', 4, 3);
INSERT INTO `train_data` VALUES (3, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}]', 142, 'false', 6, 3);
INSERT INTO `train_data` VALUES (4, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}]', 97, 'true', 7, 3);
INSERT INTO `train_data` VALUES (5, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}]', 125, 'false', 8, 3);
INSERT INTO `train_data` VALUES (6, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 7, \"color\": 0}]', 82, 'true', 9, 3);
INSERT INTO `train_data` VALUES (7, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 7, \"color\": 0}, {\"x\": 4, \"y\": 7, \"color\": 1}]', 67, 'false', 10, 3);
INSERT INTO `train_data` VALUES (8, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 7, \"color\": 0}, {\"x\": 4, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 8, \"color\": 0}]', 113, 'true', 11, 3);
INSERT INTO `train_data` VALUES (9, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 7, \"color\": 0}, {\"x\": 4, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 4, \"color\": 1}]', 124, 'false', 12, 3);
INSERT INTO `train_data` VALUES (10, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 7, \"color\": 0}, {\"x\": 4, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 4, \"color\": 1}, {\"x\": 6, \"y\": 9, \"color\": 0}]', 99, 'true', 13, 3);
INSERT INTO `train_data` VALUES (11, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 7, \"color\": 0}, {\"x\": 4, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 4, \"color\": 1}, {\"x\": 6, \"y\": 9, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}]', 155, 'false', 14, 3);
INSERT INTO `train_data` VALUES (12, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 6, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 6, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 7, \"color\": 0}, {\"x\": 4, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 4, \"color\": 1}, {\"x\": 6, \"y\": 9, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 5, \"y\": 10, \"color\": 0}]', 85, 'true', 15, 3);
INSERT INTO `train_data` VALUES (13, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}]', 113, 'true', 1, 5);
INSERT INTO `train_data` VALUES (14, 'false', '[{\"x\": 9, \"y\": 6, \"color\": 0}]', 141, 'false', 1, 6);
INSERT INTO `train_data` VALUES (15, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}]', 128, 'false', 1, 4);
INSERT INTO `train_data` VALUES (16, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}]', 113, 'true', 2, 4);
INSERT INTO `train_data` VALUES (17, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}]', 157, 'false', 3, 4);
INSERT INTO `train_data` VALUES (18, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}]', 140, 'true', 4, 4);
INSERT INTO `train_data` VALUES (19, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}]', 143, 'false', 5, 4);
INSERT INTO `train_data` VALUES (20, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}]', 210, 'true', 6, 4);
INSERT INTO `train_data` VALUES (21, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}]', 158, 'false', 7, 4);
INSERT INTO `train_data` VALUES (22, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}]', 155, 'true', 8, 4);
INSERT INTO `train_data` VALUES (23, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}]', 173, 'false', 9, 4);
INSERT INTO `train_data` VALUES (24, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}, {\"x\": 12, \"y\": 8, \"color\": 1}]', 188, 'true', 10, 4);
INSERT INTO `train_data` VALUES (25, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}, {\"x\": 12, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 7, \"color\": 0}]', 142, 'false', 11, 4);
INSERT INTO `train_data` VALUES (26, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}, {\"x\": 12, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 7, \"color\": 0}, {\"x\": 11, \"y\": 5, \"color\": 1}]', 170, 'true', 12, 4);
INSERT INTO `train_data` VALUES (27, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}, {\"x\": 12, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 7, \"color\": 0}, {\"x\": 11, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 6, \"color\": 0}]', 171, 'false', 13, 4);
INSERT INTO `train_data` VALUES (28, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}, {\"x\": 12, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 7, \"color\": 0}, {\"x\": 11, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 6, \"color\": 0}, {\"x\": 12, \"y\": 5, \"color\": 1}]', 185, 'true', 14, 4);
INSERT INTO `train_data` VALUES (29, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}, {\"x\": 12, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 7, \"color\": 0}, {\"x\": 11, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 6, \"color\": 0}, {\"x\": 12, \"y\": 5, \"color\": 1}, {\"x\": 13, \"y\": 5, \"color\": 0}]', 200, 'false', 15, 4);
INSERT INTO `train_data` VALUES (30, 'false', '[{\"x\": 8, \"y\": 8, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 10, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 5, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 14, \"y\": 0, \"color\": 1}, {\"x\": 10, \"y\": 8, \"color\": 0}, {\"x\": 10, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 8, \"color\": 0}, {\"x\": 12, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 7, \"color\": 0}, {\"x\": 11, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 6, \"color\": 0}, {\"x\": 12, \"y\": 5, \"color\": 1}, {\"x\": 13, \"y\": 5, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}]', 125, 'true', 16, 4);
INSERT INTO `train_data` VALUES (36, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}]', 112, 'true', 1, 5);
INSERT INTO `train_data` VALUES (37, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}]', 113, 'false', 2, 5);
INSERT INTO `train_data` VALUES (38, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}]', 127, 'true', 3, 5);
INSERT INTO `train_data` VALUES (39, 'false', '[{\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 7, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}]', 128, 'false', 4, 5);
INSERT INTO `train_data` VALUES (40, 'false', '[]', 112, 'false', 0, 3);
INSERT INTO `train_data` VALUES (41, 'false', '[]', 113, 'false', 0, 4);
INSERT INTO `train_data` VALUES (42, 'false', '[]', 157, 'false', 0, 4);
INSERT INTO `train_data` VALUES (43, 'false', '[]', 127, 'false', 0, 2);
INSERT INTO `train_data` VALUES (44, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}]', 113, 'true', 1, 5);
INSERT INTO `train_data` VALUES (45, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}]', 128, 'false', 2, 5);
INSERT INTO `train_data` VALUES (46, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}]', 127, 'true', 3, 5);
INSERT INTO `train_data` VALUES (47, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}]', 142, 'false', 4, 5);
INSERT INTO `train_data` VALUES (48, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}]', 141, 'true', 5, 5);
INSERT INTO `train_data` VALUES (49, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 10, \"y\": 6, \"color\": 1}]', 156, 'false', 6, 5);
INSERT INTO `train_data` VALUES (50, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 10, \"y\": 6, \"color\": 1}, {\"x\": 10, \"y\": 5, \"color\": 0}]', 155, 'true', 7, 5);
INSERT INTO `train_data` VALUES (51, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 10, \"y\": 6, \"color\": 1}, {\"x\": 10, \"y\": 5, \"color\": 0}, {\"x\": 11, \"y\": 5, \"color\": 1}]', 170, 'false', 8, 5);
INSERT INTO `train_data` VALUES (52, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 8, \"y\": 7, \"color\": 0}, {\"x\": 9, \"y\": 7, \"color\": 1}, {\"x\": 9, \"y\": 6, \"color\": 0}, {\"x\": 10, \"y\": 6, \"color\": 1}, {\"x\": 10, \"y\": 5, \"color\": 0}, {\"x\": 11, \"y\": 5, \"color\": 1}, {\"x\": 11, \"y\": 4, \"color\": 0}]', 169, 'true', 9, 5);
INSERT INTO `train_data` VALUES (53, 'false', '[]', 113, 'false', 0, 5);
INSERT INTO `train_data` VALUES (54, 'false', '[]', 128, 'false', 0, 5);
INSERT INTO `train_data` VALUES (55, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}]', 113, 'false', 1, 6);
INSERT INTO `train_data` VALUES (56, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}]', 128, 'true', 2, 6);
INSERT INTO `train_data` VALUES (57, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}]', 143, 'false', 3, 6);
INSERT INTO `train_data` VALUES (58, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 9, \"color\": 1}]', 129, 'true', 4, 6);
INSERT INTO `train_data` VALUES (59, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 9, \"color\": 1}, {\"x\": 7, \"y\": 7, \"color\": 0}]', 112, 'false', 5, 6);
INSERT INTO `train_data` VALUES (60, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 9, \"color\": 1}, {\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 7, \"color\": 1}]', 127, 'true', 6, 6);
INSERT INTO `train_data` VALUES (61, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 9, \"color\": 1}, {\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 6, \"color\": 0}]', 111, 'false', 7, 6);
INSERT INTO `train_data` VALUES (62, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 9, \"color\": 1}, {\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 6, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}]', 126, 'true', 8, 6);
INSERT INTO `train_data` VALUES (63, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 9, \"color\": 1}, {\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 6, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 7, \"y\": 5, \"color\": 0}]', 110, 'false', 9, 6);
INSERT INTO `train_data` VALUES (64, 'false', '[{\"x\": 7, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 8, \"color\": 1}, {\"x\": 9, \"y\": 8, \"color\": 0}, {\"x\": 8, \"y\": 9, \"color\": 1}, {\"x\": 7, \"y\": 7, \"color\": 0}, {\"x\": 8, \"y\": 7, \"color\": 1}, {\"x\": 7, \"y\": 6, \"color\": 0}, {\"x\": 8, \"y\": 6, \"color\": 1}, {\"x\": 7, \"y\": 5, \"color\": 0}, {\"x\": 8, \"y\": 5, \"color\": 1}]', 125, 'true', 10, 6);

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `password` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `email` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `registration_time` datetime NOT NULL ON UPDATE CURRENT_TIMESTAMP,
  `avatar` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `state` enum('online','isPK','seek') CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT 'online',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username`) USING BTREE,
  UNIQUE INDEX `email`(`email`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES (1, 'admin', 'admin', '3048679410@qq.com', '2024-06-16 08:48:28', 'image/login.jpg', 'online');
INSERT INTO `user` VALUES (2, 'coffeecat', '123456789', '202112120@stu.neu.edu.cn', '2024-06-16 08:48:28', 'image/avatar.jpg', 'online');

SET FOREIGN_KEY_CHECKS = 1;
