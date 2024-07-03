import React, { useState, useEffect, useCallback } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Paper, CardActionArea, CardMedia, TableContainer, Table, TableBody, TableHead, TableRow, TableCell, Button, CircularProgress, LinearProgress } from "@material-ui/core";
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
import axios from "axios";
import Clear from '@material-ui/icons/Clear';
import cblogo from "./cblogo.PNG";
import { DropzoneArea } from 'material-ui-dropzone';
import confetti from "canvas-confetti";

const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(theme.palette.common.white),
    backgroundColor: theme.palette.common.white,
    '&:hover': {
      backgroundColor: '#ffffff7a',
    },
  },
}))(Button);

const useStyles = makeStyles((theme) => ({
  root: {
    position: 'relative',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  videoBackground: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    zIndex: -1,
    opacity:"90%"
  },
  appbar: {
    background: 'transparent',
    boxShadow: 'none',
    color: 'white',
  },
  toolbar: {
    display: 'flex',
    color: 'white',
    background:'black',
    opacity:"60%",
    justifyContent: 'space-between',
  },
  logo: {
    width: '50px',
  },
  mainContent: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    color: 'white',
  },
  card: {
    backgroundColor: 'rgba(125, 120,105, 0.7)',
    borderRadius: '20px',
    padding: theme.spacing(4),
    maxWidth: '600px',
    width: '100%',
    boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1)',
  },
  title: {
    color:"white",
    fontSize: '2.5rem',
    fontWeight: 'bold',
    marginBottom: theme.spacing(2),
  },
  dropzoneText: {
    fontSize: "18px",
    fontWeight: "bold",
    color: "black",
    textAlign: "center",
  },
  media: {
    height: '300px',
    borderRadius: '20px',
    marginBottom: theme.spacing(2),
  },
  detail: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  table: {
    marginTop: theme.spacing(2),
    '& td, & th': {
      color: 'black',
    },
  },
  clearButton: {
    marginTop: theme.spacing(2),
    backgroundColor: '#f44336',
    color: '#fff',
    '&:hover': {
      backgroundColor: '#d32f2f',
    },
  },
  confettiContainer: {
    position: 'relative',
  },
  copyright: {
    textAlign: 'center',
    padding: theme.spacing(3),
    color: 'white',
    background:'black',
    fontWeight: 'bold',
    opacity:"60%",
    marginTop: 'auto',
  },
}));

export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [image, setImage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  let confidence = 0;

  const sendFile = useCallback(async () => {
    if (selectedFile) {
      setIsLoading(true);
      let formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const response = await axios({
          method: "post",
          url: "http://localhost:8000/predict",
          data: formData,
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const progressPercent = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(progressPercent);
          },
        });

        if (response.status === 200) {
          setData(response.data);

          if (response.data.class === "Normal") {
            confetti({
              particleCount: 200,
              spread: 160,
            });
          }
        }
      } catch (error) {
        console.error("Error uploading file:", error);
        // Handle error appropriately
      } finally {
        setIsLoading(false);
        setProgress(0);
      }
    }
  }, [selectedFile]);

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) {
      return;
    }

    setIsLoading(true);
    sendFile();
  }, [preview, sendFile]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(null);
      setImage(false);
      setData(null);
      return;
    }

    setSelectedFile(files[0]);
    setData(null);
    setImage(true);
  };

  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  return (
    <div className={classes.root}>
      <video className={classes.videoBackground} autoPlay loop muted>
        <source src="https://cdn.prod.website-files.com/6645b8579425307b143acbcc%2F665f00db1b584f61ca7fc4d7_evergreen_tablet_x2-transcode.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar className={classes.toolbar}>
          <Typography variant="h6" noWrap>
            Smart: Hallux Valgus Classification
          </Typography>
          <Avatar src={cblogo} className={classes.logo} />
        </Toolbar>
      </AppBar>
      <div className={classes.mainContent}>
        <Card className={classes.card}>
          <Typography variant="h4" className={classes.title}>
            Upload Foot Image
          </Typography>
          <CardContent>
            {!image ? (
              <DropzoneArea
                acceptedFiles={['image/*']}
                dropzoneText={"Drag and drop an image here or click"}
                onChange={onSelectFile}
                showPreviewsInDropzone={false}
                showFileNamesInPreview={false}
                filesLimit={1}
                maxFileSize={3000000}
                dropzoneClass={classes.dropzone}
                dropzoneParagraphClass={classes.dropzoneText}
                Icon={CloudUploadIcon}
                IconComponent={() => <CloudUploadIcon className={classes.uploadIcon} />}
              />
            ) : (
              <>
                <CardActionArea>
                  <CardMedia
                    className={classes.media}
                    image={preview}
                    component="img"
                    title="Uploaded Image"
                  />
                </CardActionArea>
                {data && (
                  <CardContent className={classes.detail}>
                    <TableContainer className={classes.tableContainer} component={Paper}>
                      <Table className={classes.table} aria-label="simple table">
                        <TableHead>
                          <TableRow>
                            <TableCell>Label</TableCell>
                            <TableCell align="right">Confidence</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          <TableRow>
                            <TableCell component="th" scope="row">
                              {data.class}
                            </TableCell>
                            <TableCell align="right">{confidence}%</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                    {isLoading && (
                      <React.Fragment>
                        <CircularProgress color="secondary" className={classes.loader} />
                        <LinearProgress className={classes.progress} variant="determinate" value={progress} />
                      </React.Fragment>
                    )}
                  </CardContent>
                )}
                <ColorButton
                  variant="contained"
                  color="primary"
                  size="large"
                  className={classes.clearButton}
                  onClick={clearData}
                  startIcon={<Clear />}
                >
                  Clear
                </ColorButton>
              </>
            )}
          </CardContent>
        </Card>
      </div>
      <Typography  variant="body2" className={classes.copyright}>
      ©2024 Department of Orthopaedic Surgery (PGIMER Chandigarh) 
      Design and developed by Amit Kumar (Project Research Scientist -I)
    
      </Typography>
    </div>
  );
};
