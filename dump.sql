/*
SQLyog Community Edition- MySQL GUI v7.15 
MySQL - 5.5.29 : Database - wpdb
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`wpdb` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `wpdb`;

/*Table structure for table `points` */

DROP TABLE IF EXISTS `points`;

CREATE TABLE `points` (
  `id` int(110) NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `points` char(100) NOT NULL,
  `current_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `p` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;

/*Data for the table `points` */

LOCK TABLES `points` WRITE;

insert  into `points`(`id`,`username`,`points`,`current_time`,`p`) values (5,'sanjay','1,2,2,2,0,1,1,0,1,1,1,2,2,1,2,1,2,1,1,1,1,0','2025-02-07 11:45:53','Student Performance is Medium'),(6,'sanjay','1,2,2,2,0,1,1,0,1,1,1,2,2,1,2,1,2,1,1,1,1,0','2025-02-07 11:47:05','Student Performance is Medium'),(7,'sanjay','2,2,2,2,0,1,2,1,0,1,2,2,2,2,0,0,2,1,2,0,1,0','2025-02-07 13:00:13','Student Performance is High'),(8,'sanjay','1,2,2,2,2,2,1,1,2,2,2,2,2,2,2,1,1,2,1,2,1,2','2025-02-07 13:43:01','Student Performance is Medium');

UNLOCK TABLES;

/*Table structure for table `user` */

DROP TABLE IF EXISTS `user`;

CREATE TABLE `user` (
  `id` int(100) NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=latin1;

/*Data for the table `user` */

LOCK TABLES `user` WRITE;

insert  into `user`(`id`,`username`,`email`,`password`) values (3,'sanjay','123@gmail.com','123456'),(4,'sanjay1','123@gmail.com','123456');

UNLOCK TABLES;

/*Table structure for table `user1` */

DROP TABLE IF EXISTS `user1`;

CREATE TABLE `user1` (
  `id` int(100) NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=latin1;

/*Data for the table `user1` */

LOCK TABLES `user1` WRITE;

insert  into `user1`(`id`,`username`,`email`,`password`) values (5,'sanjay','pamalasanjaykumar@gmail.com','123456');

UNLOCK TABLES;

/*Table structure for table `vs` */

DROP TABLE IF EXISTS `vs`;

CREATE TABLE `vs` (
  `id` int(110) NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `link` char(100) NOT NULL,
  `current_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `score` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=29 DEFAULT CHARSET=latin1;

/*Data for the table `vs` */

LOCK TABLES `vs` WRITE;

insert  into `vs`(`id`,`username`,`link`,`current_time`,`score`) values (26,'sanjay','sanjay_20250207_120230.mp4','2025-02-07 12:03:46','{\'blinks\': 7, \'yawns\': 135, \'face_angle_changes\': 2}'),(27,'sanjay','sanjay_20250207_122918.mp4','2025-02-07 12:30:35','{\'blinks\': 7, \'yawns\': 135, \'face_angle_changes\': 2}'),(28,'sanjay','sanjay_20250207_134316.mp4','2025-02-07 13:44:32','{\'blinks\': 7, \'yawns\': 135, \'face_angle_changes\': 2}');

UNLOCK TABLES;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
