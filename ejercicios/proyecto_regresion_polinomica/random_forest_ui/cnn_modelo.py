"""CNN de regresión 1D y 2D con particiones train/validación/prueba."""
import copy
import io
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from modelo import metricas


def crear_red(dimension, canales, config):
    conv = nn.Conv1d if dimension == 1 else nn.Conv2d
    adaptive = nn.AdaptiveAvgPool1d if dimension == 1 else nn.AdaptiveAvgPool2d
    pool = (nn.MaxPool1d if dimension == 1 else nn.MaxPool2d) if config['pooling'] == 'max' else (nn.AvgPool1d if dimension == 1 else nn.AvgPool2d)
    activacion = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}[config['activation']]
    capas = []
    for _ in range(config['layers']):
        capas += [conv(canales, config['filters'], config['kernel_size'], stride=config['stride'],
                       padding=config['dilation']*(config['kernel_size']//2), dilation=config['dilation']), activacion()]
        if config['pooling'] != 'none':
            capas.append(pool(2, ceil_mode=True))
        capas.append(nn.Dropout(config['dropout']))
        canales = config['filters']
    capas += [adaptive(1), nn.Flatten(), nn.Linear(canales, config['hidden_units']), activacion(),
              nn.Dropout(config['dropout']), nn.Linear(config['hidden_units'], 1)]
    return nn.Sequential(*capas)


def ejemplo_1d():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2*np.pi, 32)
    amplitud = rng.uniform(.2, 3., 240)
    x = amplitud[:, None]*np.sin(t)[None, :] + rng.normal(0, .08, (240, 32))
    frame = pd.DataFrame(x, columns=[f'senal_{i:02d}' for i in range(32)])
    frame['objetivo'] = amplitud*5 + rng.normal(0, .1, len(x))
    return frame


def ejemplo_2d():
    rng = np.random.default_rng(42)
    nivel = rng.uniform(.1, .9, 160)
    x = np.clip(nivel[:, None, None, None] + rng.normal(0, .04, (160, 1, 16, 16)), 0, 1).astype('float32')
    return x, (nivel*10).astype('float32'), [f'imagen_{i:03d}.png' for i in range(len(x))]


def zip_ejemplo():
    x, y, nombres = ejemplo_2d()
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('labels.csv', pd.DataFrame({'file': nombres, 'target': y}).to_csv(index=False))
        for image, nombre in zip(x, nombres):
            buffer = io.BytesIO()
            Image.fromarray((image[0]*255).astype('uint8')).save(buffer, format='PNG')
            archive.writestr(nombre, buffer.getvalue())
    return output.getvalue()


def leer_imagenes(raw, tamano, color):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > 2000 or sum(info.file_size for info in entries) > 100*1024*1024:
            raise ValueError('El ZIP admite como máximo 2.000 entradas y 100 MB descomprimidos.')
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError('El ZIP contiene nombres duplicados.')
        frame = pd.read_csv(io.BytesIO(archive.read('labels.csv')))
        if not {'file', 'target'} <= set(frame.columns) or not 20 <= len(frame) <= 1000:
            raise ValueError('labels.csv debe tener file y target, con entre 20 y 1.000 imágenes.')
        if frame['file'].duplicated().any():
            raise ValueError('Cada imagen debe aparecer una sola vez en labels.csv.')
        targets = pd.to_numeric(frame['target'], errors='raise').to_numpy(dtype='float32')
        if not np.isfinite(targets).all():
            raise ValueError('Los objetivos de labels.csv deben ser números finitos.')
        imagenes = []
        for nombre in frame['file']:
            if not isinstance(nombre, str) or not nombre.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                raise ValueError('Solo se admiten imágenes PNG, JPG o BMP con rutas relativas dentro del ZIP.')
            with Image.open(io.BytesIO(archive.read(nombre))) as image:
                if image.width*image.height > 16_000_000:
                    raise ValueError('Cada imagen admite como máximo 16 millones de píxeles.')
                valores = np.asarray(image.convert(color).resize((tamano, tamano)), dtype='float32')/255
                imagenes.append(valores[None, :, :] if color == 'L' else valores.transpose(2, 0, 1))
        return np.stack(imagenes), targets, frame['file'].tolist()


def entrenar_cnn(x, y, dimension, config, prueba=.2, semilla=42, cronologico=False, progreso=None, particion=None):
    x, y = np.asarray(x, dtype='float32'), np.asarray(y, dtype='float32')
    if len(x) < 20 or len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError('Se requieren al menos 20 muestras y objetivos numéricos finitos.')
    if x.ndim != dimension+2:
        raise ValueError('Forma de entrada incompatible con la dimensión de la CNN.')
    torch.set_num_threads(2)
    torch.manual_seed(semilla)
    indices = np.arange(len(y))
    if particion is None:
        development, test = train_test_split(indices, test_size=prueba, random_state=semilla, shuffle=not cronologico)
    else:
        development, test = (np.asarray(i, dtype=int) for i in particion)
        if len(development) < 4 or len(test) < 2 or np.intersect1d(development, test).size:
            raise ValueError('Particiones CNN insuficientes o solapadas.')
    train, validation = train_test_split(development, test_size=.2, random_state=semilla, shuffle=not cronologico)
    axes = (0,) if dimension == 1 else (0, 2, 3)
    limpio = np.where(np.isfinite(x), x, np.nan)
    mediana = np.nanmedian(limpio[train], axis=axes, keepdims=True)
    if not np.isfinite(mediana).all():
        raise ValueError('Hay posiciones o canales sin datos válidos en entrenamiento.')
    limpio = np.where(np.isfinite(limpio), limpio, mediana)
    media = limpio[train].mean(axis=axes, keepdims=True)
    escala = limpio[train].std(axis=axes, keepdims=True)
    escala = np.where(escala < 1e-7, 1, escala)
    y_media, y_escala = float(y[train].mean()), max(float(y[train].std()), 1e-7)
    xt = torch.tensor((limpio-media)/escala)
    yt = torch.tensor((y-y_media)/y_escala).reshape(-1, 1)
    modelo = crear_red(dimension, x.shape[1], config)
    optimizadores = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW, 'SGD': torch.optim.SGD}
    extra = {'momentum': config['momentum']} if config['optimizer'] == 'SGD' else {}
    optimizador = optimizadores[config['optimizer']](modelo.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], **extra)
    perdida = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss, 'Huber': nn.SmoothL1Loss}[config['loss']]()
    generador = torch.Generator().manual_seed(semilla)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt[train], yt[train]),
                                         batch_size=config['batch_size'], shuffle=True, generator=generador)
    mejor, mejor_estado, sin_mejora, historia = float('inf'), None, 0, []
    for epoch in range(config['epochs']):
        modelo.train()
        total = 0.
        for xb, yb in loader:
            optimizador.zero_grad()
            error = perdida(modelo(xb), yb)
            if not torch.isfinite(error):
                raise ValueError('El entrenamiento diverge. Reduce learning_rate o revisa los datos.')
            error.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 10.)
            optimizador.step()
            total += float(error.detach())*len(xb)
        modelo.eval()
        with torch.no_grad():
            valid_loss = float(perdida(modelo(xt[validation]), yt[validation]))
        if not np.isfinite(valid_loss):
            raise ValueError('La pérdida de validación no es finita; revisa learning_rate y los datos.')
        historia.append({'Época': epoch+1, 'Entrenamiento': total/len(train), 'Validación': valid_loss})
        if valid_loss < mejor-config['min_delta']:
            mejor, mejor_estado, sin_mejora = valid_loss, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
        if progreso:
            progreso((epoch+1)/config['epochs'])
        if config['patience'] and sin_mejora >= config['patience']:
            break
    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        pred = modelo(xt[test]).numpy().ravel()*y_escala+y_media
        train_pred = modelo(xt[train]).numpy().ravel()*y_escala+y_media
    resultados = pd.DataFrame({'indice': test, 'observado': y[test], 'predicho': pred, 'residuo': y[test]-pred})
    paquete = {'format_version': 1, 'dimension': dimension, 'channels': x.shape[1], 'input_shape': list(x.shape[1:]),
               'config': config, 'state_dict': modelo.state_dict(), 'median': mediana.tolist(), 'mean': media.tolist(),
               'scale': escala.tolist(), 'target_mean': y_media, 'target_scale': y_escala,
               'indices': {'train': train.tolist(), 'validation': validation.tolist(), 'test': test.tolist()}}
    return paquete, resultados, metricas(y[test], pred), metricas(y[train], train_pred), pd.DataFrame(historia)


def predecir_cnn(paquete, x):
    modelo = crear_red(paquete['dimension'], paquete['channels'], paquete['config'])
    modelo.load_state_dict(paquete['state_dict'])
    modelo.eval()
    x = np.asarray(x, dtype='float32')
    if list(x.shape[1:]) != paquete['input_shape']:
        raise ValueError('Las dimensiones de las nuevas muestras deben coincidir con input_shape del modelo.')
    x = np.where(np.isfinite(x), x, np.asarray(paquete['median'], dtype='float32'))
    x = (x-np.asarray(paquete['mean'], dtype='float32'))/np.asarray(paquete['scale'], dtype='float32')
    with torch.no_grad():
        return modelo(torch.tensor(x)).numpy().ravel()*paquete['target_scale']+paquete['target_mean']
